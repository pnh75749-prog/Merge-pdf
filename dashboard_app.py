from __future__ import annotations

import hmac
import json
import os
import tempfile
import time
import uuid
from io import BytesIO
from decimal import Decimal
from functools import lru_cache
from pathlib import Path, PurePosixPath
import re
from typing import TYPE_CHECKING

from flask import Flask, jsonify, render_template_string, request
from pypdf import PdfReader, PdfWriter

if TYPE_CHECKING:
  import numpy as np
  from rapidocr_onnxruntime import RapidOCR

app = Flask(__name__)
ocr_engine = None
app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024

MAX_PDF_BYTES = int(os.getenv("MAX_PDF_BYTES", str(500 * 1024 * 1024)))
MAX_WORKSPACE_BYTES = int(os.getenv("MAX_WORKSPACE_BYTES", str(2 * 1024 * 1024 * 1024)))
MAX_WORKSPACE_FILES = 100

HTML_PAGE = """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>PDF Workshop</title>
    <style>
      * { box-sizing: border-box; }
      body {
        margin: 0;
        font-family: Arial, sans-serif;
        background: linear-gradient(135deg, #0f172a, #1e293b);
        color: #e2e8f0;
      }
      .container {
        max-width: 900px;
        margin: 60px auto;
        background: rgba(15, 23, 42, 0.8);
        border: 1px solid #334155;
        border-radius: 18px;
        box-shadow: 0 20px 50px rgba(0, 0, 0, 0.35);
        padding: 32px;
      }
      h1 {
        margin-top: 0;
        font-size: 2rem;
      }
      .dropzone {
        border: 2px dashed #38bdf8;
        border-radius: 16px;
        padding: 36px 20px;
        text-align: center;
        background: rgba(14, 116, 144, 0.08);
        cursor: pointer;
        transition: all 0.2s ease;
        margin-bottom: 20px;
      }
      .dropzone.dragover {
        background: rgba(56, 189, 248, 0.14);
        transform: scale(1.01);
      }
      .hidden-input {
        display: none;
      }
      .button-row {
        display: flex;
        gap: 12px;
        flex-wrap: wrap;
        align-items: center;
      }
      button {
        border: none;
        border-radius: 10px;
        padding: 12px 18px;
        font-size: 1rem;
        cursor: pointer;
        background: #38bdf8;
        color: #082f49;
        font-weight: 700;
      }
      button:disabled {
        cursor: not-allowed;
        opacity: 0.55;
      }
      button.secondary {
        background: #e2e8f0;
        color: #0f172a;
      }
      .summary {
        margin-top: 18px;
        color: #bae6fd;
        min-height: 28px;
      }
      .progress-panel {
        margin-top: 18px;
      }
      .output-location {
        overflow-wrap: anywhere;
        color: #fbbf24;
      }
      .progress-panel[hidden] {
        display: none;
      }
      .progress-labels {
        display: flex;
        justify-content: space-between;
        gap: 12px;
        margin-bottom: 8px;
        font-size: 0.9rem;
      }
      .progress-track {
        position: relative;
        height: 12px;
        overflow: hidden;
        border-radius: 6px;
        background: #334155;
      }
      .progress-bar {
        position: relative;
        width: 0;
        height: 100%;
        border-radius: inherit;
        background: #38bdf8;
        transition: width 0.25s ease;
      }
      .progress-track.is-active::after {
        position: absolute;
        top: 0;
        left: 0;
        width: 28%;
        height: 100%;
        border-radius: inherit;
        background: #fbbf24;
        content: "";
        animation: progress-sweep 1.1s ease-in-out infinite;
      }
      @keyframes progress-sweep {
        from { transform: translateX(-110%); }
        to { transform: translateX(360%); }
      }
      ul {
        padding-left: 20px;
      }
      .folder-list {
        margin-top: 16px;
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 12px 16px;
      }
      .folder-list li.is-complete {
        width: fit-content;
        border-radius: 4px;
        background: rgba(34, 197, 94, 0.14);
        color: #bbf7d0;
        font-weight: 700;
      }
      .folder-list li.is-complete::after {
        color: #86efac;
        content: "  completed";
        font-size: 0.85em;
        font-weight: 400;
      }
      .folder-list li.is-failed {
        width: fit-content;
        border-radius: 4px;
        background: rgba(248, 113, 113, 0.14);
        color: #fecaca;
      }
      .folder-list li.is-failed::after {
        color: #fca5a5;
        content: "  skipped";
        font-size: 0.85em;
      }
      .folder-list li.rename-file.is-failed[data-status="failed"]::after {
        content: "  failed";
      }
      .manual-rename-button {
        margin-left: 8px;
        padding: 5px 8px;
        font-size: 0.85rem;
      }
      .completion-dialog {
        width: min(440px, calc(100vw - 32px));
        border: 1px solid #475569;
        border-radius: 8px;
        padding: 24px;
        background: #0f172a;
        color: #e2e8f0;
      }
      .completion-dialog::backdrop {
        background: rgba(2, 6, 23, 0.72);
      }
      .completion-dialog h2 {
        margin-top: 0;
      }
      .completion-dialog p {
        overflow-wrap: anywhere;
        line-height: 1.5;
      }
      :root {
        color-scheme: light;
        font-family: "Segoe UI", Inter, system-ui, sans-serif;
        color: #171923;
        background: #e9ecf4;
        font-synthesis: none;
      }
      body {
        min-width: 320px;
        background:
          radial-gradient(ellipse at 70% -20%, rgba(255, 255, 255, 0.9), transparent 48%),
          #e9ecf4;
        color: #171923;
        font-family: "Segoe UI", Inter, system-ui, sans-serif;
      }
      button {
        font: inherit;
      }
      button:focus-visible,
      input:focus-visible {
        outline: 3px solid rgba(91, 125, 207, 0.45);
        outline-offset: 2px;
      }
      .app-shell {
        display: flex;
        min-height: 100vh;
      }
      .sidebar {
        position: sticky;
        top: 0;
        display: flex;
        width: 232px;
        height: 100vh;
        flex: 0 0 232px;
        flex-direction: column;
        gap: 28px;
        padding: 26px 18px;
        border-right: 1px solid rgba(255, 255, 255, 0.85);
        background: rgba(247, 248, 252, 0.72);
      }
      .brand {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 0 8px;
        color: inherit;
        font-size: 15px;
        font-weight: 700;
        text-decoration: none;
      }
      .brand-mark {
        display: grid;
        width: 38px;
        height: 38px;
        place-items: center;
        border-radius: 13px;
        background: #171923;
        color: #fff;
        font-size: 11px;
        letter-spacing: -0.05em;
      }
      .nav-caption {
        padding: 0 10px;
        color: #858b9a;
        font-size: 10px;
        font-weight: 700;
        letter-spacing: 0.13em;
        text-transform: uppercase;
      }
      .sidebar-nav {
        display: grid;
        gap: 6px;
      }
      .nav-link {
        display: flex;
        min-height: 43px;
        align-items: center;
        gap: 12px;
        border: 0;
        border-radius: 14px;
        padding: 0 12px;
        background: transparent;
        color: #697080;
        cursor: pointer;
        font-size: 13px;
        font-weight: 600;
        text-align: left;
      }
      .nav-link:hover {
        background: rgba(255, 255, 255, 0.72);
      }
      .nav-link[aria-current="page"] {
        background: #171923;
        color: #fff;
        box-shadow: 0 6px 16px rgba(17, 19, 24, 0.12);
      }
      .nav-icon {
        width: 19px;
        font-size: 17px;
        text-align: center;
      }
      .sidebar-note {
        margin-top: auto;
        border: 1px solid rgba(255, 255, 255, 0.95);
        border-radius: 18px;
        padding: 14px;
        background: rgba(255, 255, 255, 0.55);
        color: #777e8d;
        font-size: 11px;
        line-height: 1.55;
      }
      .sidebar-note strong {
        display: block;
        margin-bottom: 3px;
        color: #272a34;
        font-size: 12px;
      }
      .main-content {
        width: min(100%, 1480px);
        min-width: 0;
        margin: 0 auto;
        padding: 24px clamp(18px, 4vw, 56px) 48px;
      }
      .topbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 16px;
        margin-bottom: 28px;
      }
      .breadcrumbs {
        display: flex;
        align-items: center;
        gap: 9px;
        color: #7d8391;
        font-size: 12px;
      }
      .breadcrumbs strong {
        color: #242731;
        font-weight: 600;
      }
      .top-actions {
        display: flex;
        align-items: center;
        gap: 12px;
      }
      .local-badge {
        display: inline-flex;
        align-items: center;
        gap: 7px;
        color: #646b79;
        font-size: 11px;
      }
      .local-badge::before {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: #51a77c;
        content: "";
      }
      .button {
        display: inline-flex;
        min-height: 40px;
        align-items: center;
        justify-content: center;
        gap: 8px;
        border: 1px solid transparent;
        border-radius: 999px;
        padding: 0 17px;
        background: #171923;
        color: #fff;
        cursor: pointer;
        font-size: 12px;
        font-weight: 650;
        transition: transform 0.15s ease, opacity 0.15s ease, background 0.15s ease;
      }
      .button:hover:not(:disabled) {
        transform: translateY(-1px);
        background: #30333e;
      }
      .button:disabled {
        cursor: not-allowed;
        opacity: 0.46;
      }
      .button.secondary {
        border-color: rgba(215, 219, 229, 0.9);
        background: rgba(255, 255, 255, 0.66);
        color: #333744;
      }
      .button.secondary:hover:not(:disabled) {
        background: #fff;
      }
      .button.danger {
        background: rgba(232, 115, 95, 0.12);
        color: #bd5848;
      }
      .button.small {
        min-height: 34px;
        padding: 0 13px;
        font-size: 11px;
      }
      .page {
        display: none;
      }
      .page.is-active {
        display: block;
        animation: page-enter 0.22s ease-out;
      }
      @keyframes page-enter {
        from { opacity: 0.55; transform: translateY(4px); }
        to { opacity: 1; transform: translateY(0); }
      }
      .page-heading {
        display: flex;
        align-items: flex-end;
        justify-content: space-between;
        gap: 18px;
        margin: 0 0 23px;
      }
      .eyebrow {
        margin: 0 0 7px;
        color: #737b8b;
        font-size: 12px;
      }
      h1,
      h2,
      h3,
      p {
        overflow-wrap: anywhere;
      }
      .page-heading h1 {
        margin: 0;
        font-size: clamp(27px, 3.2vw, 39px);
        font-weight: 650;
        letter-spacing: -0.045em;
      }
      .page-description {
        max-width: 610px;
        margin: 8px 0 0;
        color: #737b8b;
        font-size: 13px;
        line-height: 1.6;
      }
      .card {
        min-width: 0;
        border: 1px solid rgba(255, 255, 255, 0.95);
        border-radius: 25px;
        padding: 22px;
        background: rgba(247, 248, 252, 0.78);
        box-shadow: 0 8px 30px rgba(17, 19, 24, 0.035);
        backdrop-filter: blur(18px);
      }
      .card-heading {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
        margin: 0 0 17px;
      }
      .card-heading h2,
      .card h2 {
        margin: 0;
        font-size: 16px;
        font-weight: 650;
        letter-spacing: -0.02em;
      }
      .muted {
        color: #777e8d;
      }
     
      .step-name {
        margin: 7px 0 10px;
        font-size: 13px;
        font-weight: 650;
      }
      .step-status {
        color: #858b9a;
        font-size: 10px;
      }
      .dashboard-grid,
      .content-grid {
        display: grid;
        grid-template-columns: minmax(0, 1.8fr) minmax(260px, 0.9fr);
        gap: 18px;
      }
      .quick-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 11px;
      }
      .action-tile {
        display: flex;
        min-height: 80px;
        align-items: center;
        gap: 13px;
        border: 1px solid transparent;
        border-radius: 17px;
        padding: 14px;
        background: #e9ecf4;
        color: #292c36;
        cursor: pointer;
        text-align: left;
      }
      .action-tile:hover {
        border-color: #d7dce8;
        background: #fff;
      }
      .action-tile .tile-icon {
        display: grid;
        width: 37px;
        height: 37px;
        flex: 0 0 37px;
        place-items: center;
        border-radius: 13px;
        background: #fff;
        font-size: 17px;
      }
      .action-tile strong,
      .action-tile small {
        display: block;
      }
      .action-tile strong {
        font-size: 12px;
      }
      .action-tile small {
        margin-top: 4px;
        color: #7c8290;
        font-size: 10px;
      }
      .location-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 10px;
        margin-top: 14px;
      }
      .location-card {
        min-width: 0;
        border-radius: 16px;
        padding: 13px 14px;
        background: #f0f2f8;
      }
      .location-card h3 {
        margin: 0 0 6px;
        color: #777e8d;
        font-size: 10px;
        font-weight: 600;
      }
      .location-card p {
        margin: 0;
        color: #323641;
        font-size: 11px;
        line-height: 1.5;
        overflow-wrap: anywhere;
      }
      .job-card {
        margin-bottom: 18px;
      }
      .job-card[hidden] {
        display: none;
      }
      .job-header,
      .progress-labels {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
      }
      .progress-labels {
        margin: 19px 0 9px;
        font-size: 12px;
      }
      .progress-track {
        height: 9px;
        border-radius: 999px;
        background: #e0e4ee;
      }
      .progress-bar {
        border-radius: inherit;
        background: #6687d0;
        transition: width 0.25s ease;
      }
      .progress-track.is-active .progress-bar {
        background: linear-gradient(90deg, #6687d0, #91a9de);
      }
      .progress-track.is-active::after {
        display: none;
      }
      .job-actions {
        display: flex;
        flex-wrap: wrap;
        gap: 7px;
      }
      .job-actions .button {
        min-height: 32px;
        padding: 0 12px;
        font-size: 10px;
      }
      .stat-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 13px;
        margin-top: 18px;
        margin-bottom: 18px;
      }
      .stat-card {
        min-height: 114px;
        padding: 17px;
      }
      .stat-card p {
        margin: 0;
        color: #777e8d;
        font-size: 11px;
      }
      .stat-card strong {
        display: block;
        margin-top: 16px;
        font-size: 27px;
        font-weight: 650;
        letter-spacing: -0.04em;
      }
      .warning {
        display: flex;
        align-items: flex-start;
        gap: 11px;
        margin-top: 18px;
        border: 1px solid rgba(232, 115, 95, 0.18);
        border-radius: 17px;
        padding: 14px 16px;
        background: rgba(232, 115, 95, 0.08);
        color: #6f737d;
        font-size: 11px;
        line-height: 1.55;
      }
      .warning strong {
        color: #a44b3f;
      }
      .summary {
        min-height: 0;
        margin: -10px 0 18px;
        color: #596274;
        font-size: 12px;
        line-height: 1.5;
      }
      .summary:empty {
        display: none;
      }
      .folder-list {
        margin-top: 12px;
        border: 0;
        border-radius: 0;
        padding: 0;
        background: transparent;
      }
      .folder-list ul,
      #renameFileList ul {
        display: grid;
        gap: 8px;
        margin: 0;
        padding: 0;
        list-style: none;
      }
      .folder-list li,
      .rename-file {
        width: 100%;
        border: 1px solid #fff;
        border-radius: 15px;
        padding: 12px 14px;
        background: #f7f8fc;
        color: #343743;
        font-size: 12px;
        line-height: 1.5;
      }
      .folder-list li.is-complete {
        background: rgba(81, 167, 124, 0.1);
        color: #337b59;
      }
      .folder-list li.is-failed,
      .rename-file.is-failed {
        background: rgba(232, 115, 95, 0.1);
        color: #a44b3f;
      }
      #renameFileList > h3 {
        margin: 16px 0 9px;
        color: #555c6b;
        font-size: 12px;
        font-weight: 650;
      }
      #renameFileList > h3:first-child {
        margin-top: 0;
      }
      .rename-file {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
      }
      .rename-file.is-complete {
        background: rgba(81, 167, 124, 0.1);
        color: #337b59;
      }
      .manual-rename-button {
        flex: 0 0 auto;
        margin-left: 0;
      }
      .completion-dialog {
        border: 1px solid #fff;
        border-radius: 22px;
        background: #f7f8fc;
        color: #171923;
        box-shadow: 0 20px 80px rgba(17, 19, 24, 0.25);
      }
      .completion-dialog::backdrop {
        background: rgba(19, 22, 31, 0.38);
        backdrop-filter: blur(4px);
      }
      .completion-dialog h2 {
        font-size: 21px;
        letter-spacing: -0.03em;
      }
      @media (max-width: 900px) {
        .sidebar {
          width: 74px;
          flex-basis: 74px;
          align-items: center;
          padding: 20px 10px;
        }
        .brand {
          padding: 0;
        }
        .brand-name,
        .nav-caption,
        .nav-label,
        .sidebar-note {
          display: none;
        }
        .nav-link {
          width: 46px;
          justify-content: center;
          padding: 0;
        }
        .dashboard-grid,
        .content-grid {
          grid-template-columns: 1fr;
        }
      }
      @media (max-width: 620px) {
        .sidebar {
          position: fixed;
          z-index: 5;
          top: auto;
          right: 0;
          bottom: 0;
          left: 0;
          width: 100%;
          height: 66px;
          flex-direction: row;
          justify-content: center;
          border-top: 1px solid #fff;
          border-right: 0;
          padding: 8px 12px;
          background: rgba(247, 248, 252, 0.95);
          backdrop-filter: blur(16px);
        }
        .brand,
        .sidebar-note,
        .nav-caption {
          display: none;
        }
        .sidebar-nav {
          width: min(100%, 420px);
          grid-template-columns: repeat(3, 1fr);
          gap: 6px;
        }
        .nav-link {
          width: 100%;
          height: 48px;
          flex-direction: column;
          gap: 1px;
          border-radius: 13px;
          font-size: 9px;
        }
        .nav-label {
          display: block;
        }
        .nav-icon {
          font-size: 15px;
        }
        .main-content {
          padding: 18px 15px 84px;
        }
        .topbar {
          margin-bottom: 22px;
        }
        .local-badge {
          display: none;
        }
        .page-heading {
          align-items: flex-start;
          flex-direction: column;
        }
        .workflow-grid {
          grid-template-columns: repeat(2, minmax(0, 1fr));
        }
        .quick-grid,
        .location-grid {
          grid-template-columns: 1fr;
        }
        .stat-grid {
          grid-template-columns: 1fr;
        }
        .stat-card {
          min-height: 90px;
        }
        .stat-card strong {
          margin-top: 9px;
        }
        .card {
          border-radius: 21px;
          padding: 17px;
        }
        .rename-file {
          align-items: flex-start;
          flex-direction: column;
        }
      }
    </style>
  </head>
  <body>
    <div class="app-shell">
      <aside class="sidebar">
        <a class="brand" href="#" aria-label="PDF Workshop home">
          <span class="brand-mark">PDF</span><span class="brand-name">PDF Workshop</span>
        </a>
        <div class="nav-caption">Workspace</div>
        <nav class="sidebar-nav" aria-label="Workspace pages">
          <button class="nav-link" type="button" data-nav="dashboard" aria-current="page">
            <span class="nav-icon" aria-hidden="true">⌂</span><span class="nav-label">Dashboard</span>
          </button>
          <button class="nav-link" type="button" data-nav="merge">
            <span class="nav-icon" aria-hidden="true">▤</span><span class="nav-label">Merge PDFs</span>
          </button>
          <button class="nav-link" type="button" data-nav="rename">
            <span class="nav-icon" aria-hidden="true">✎</span><span class="nav-label">Rename PDFs</span>
          </button>
        </nav>
        <div class="sidebar-note">
          <strong>Private cloud workspace</strong>
          Uploads stay private. Original PDFs are never changed.
        </div>
      </aside>

      <main class="main-content">
        <header class="topbar">
          <div class="breadcrumbs"><span>PDF Workshop</span><span>/</span><strong id="pageTitle">Dashboard</strong></div>
          <div class="top-actions">
            <span class="local-badge">Cloud workspace</span>
            <button class="button secondary small" id="clearButton" type="button">Clear session</button>
          </div>
        </header>

        <section id="progressPanel" class="job-card card" hidden aria-live="polite">
          <div class="job-header">
            <div>
              <p class="eyebrow">Current job</p>
              <strong id="progressText">Ready</strong>
            </div>
            <div class="job-actions">
              <button class="button secondary" id="resumeButton" type="button" disabled>Resume</button>
              <button class="button secondary" id="pauseButton" type="button" disabled>Pause</button>
              <button class="button danger" id="stopButton" type="button" disabled>Stop</button>
            </div>
          </div>
          <div class="progress-labels">
            <span class="muted">Overall progress</span><span id="progressPercent">0%</span>
          </div>
          <div class="progress-track" role="progressbar" aria-label="PDF processing progress" aria-valuemin="0" aria-valuemax="100" aria-valuenow="0">
            <div id="progressBar" class="progress-bar" style="width: 0%"></div>
          </div>
        </section>
        <div id="summary" class="summary" role="status" aria-live="polite"></div>

        <section class="page is-active" data-page="dashboard">
          <div class="page-heading">
            <div>
              <p class="eyebrow">PDF Merge Workshop</p>
              <h1>Dashboard</h1>
              <p class="page-description">Upload a folder or select PDFs, then create renamed copies or merge each subfolder.</p>
            </div>
          </div>

            <div class="stat-grid">
                      <section class="card stat-card"><p>PDFs in main folder</p><strong id="statsPdfCount">0</strong></section>
                      <section class="card stat-card"><p>Subfolders ready to merge</p><strong id="statsFolderCount">0</strong></section>
                      <section class="card stat-card"><p>PDFs renamed this run</p><strong id="statsRenamedCount">0</strong></section>
             </div>

          <div class="dashboard-grid">
            <section class="card">
              <div class="card-heading"><h2>Quick actions</h2></div>
              <div class="quick-grid">
                <button class="action-tile" id="sourceButton" type="button">
                  <span class="tile-icon" aria-hidden="true">▣</span><span><strong>Upload folder</strong><small>Keep subfolder structure</small></span>
                </button>
                <button class="action-tile" id="pdfButton" type="button">
                  <span class="tile-icon" aria-hidden="true">＋</span><span><strong>Upload PDFs</strong><small>Select multiple files</small></span>
                </button>
                <button class="action-tile" id="outputButton" type="button">
                  <span class="tile-icon" aria-hidden="true">⇩</span><span><strong>Download results</strong><small>Get completed PDF outputs</small></span>
                </button>
                <button class="action-tile" type="button" data-nav="merge">
                  <span class="tile-icon" aria-hidden="true">▤</span><span><strong>Merge PDFs</strong><small>Combine files in each subfolder</small></span>
                </button>
                <button class="action-tile" type="button" data-nav="rename">
                  <span class="tile-icon" aria-hidden="true">✎</span><span><strong>Rename PDFs</strong><small>Use the ID on each first page</small></span>
                </button>
              </div>
              <div class="location-grid">
                <div class="location-card"><h3>SOURCE</h3><p id="sourceLocation">Source: No PDFs uploaded.</p></div>
                <div class="location-card"><h3>CLOUD RESULTS</h3><p id="outputLocation" aria-live="polite">Processed PDFs are stored securely for a limited time.</p><div id="downloadList"></div></div>
              </div>
              <input id="folderInput" class="hidden-input" type="file" webkitdirectory directory multiple accept="application/pdf,.pdf" />
              <input id="pdfInput" class="hidden-input" type="file" multiple accept="application/pdf,.pdf" />
            </section>

            <section class="card">
              <div class="card-heading"><h2>Current job</h2><span class="muted" style="font-size: 11px">Live status</span></div>
              <div style="display: grid; min-height: 125px; place-content: center; text-align: center">
                <div style="font-size: 30px; font-weight: 650; letter-spacing: -0.04em" id="dashboardProgress">0%</div>
                <div class="muted" style="margin-top: 4px; font-size: 11px" id="dashboardJobStatus">No job running</div>
              </div>
              <p class="muted" style="margin: 0; text-align: center; font-size: 11px">Start a merge or rename task to see its progress.</p>
            </section>
          </div>

        

          <div class="warning">
            <span aria-hidden="true">⚠</span>
            <div><strong>Original files are preserved.</strong> Merged and renamed files are saved separately in cloud storage and expire after the configured retention period.</div>
          </div>
        </section>

        <section class="page" data-page="merge">
          <div class="page-heading">
            <div>
              <p class="eyebrow">Stage 3 of 4</p>
              <h1>Merge PDFs</h1>
              <p class="page-description">Combine the PDFs in each subfolder into a separate downloadable file. Original uploads remain unchanged.</p>
            </div>
            <button class="button" id="mergeButton" type="button">Start merge</button>
          </div>
          <div class="content-grid">
            <section class="card">
              <div class="card-heading">
                <div><h2>Subfolders</h2><p class="muted" style="margin: 5px 0 0; font-size: 11px" id="mergeFolderCount">Choose a main folder to begin.</p></div>
              </div>
              <div id="folderList" class="folder-list" aria-live="polite"><p>No folder selected yet.</p></div>
            </section>
            <aside class="card">
              <div class="card-heading"><h2>Before you merge</h2></div>
              <p class="muted" style="margin: 0; font-size: 12px; line-height: 1.7">Upload a folder or select PDFs from the Dashboard. Each subfolder with PDFs is saved as its own cloud result.</p>
            </aside>
          </div>
        </section>

        <section class="page" data-page="rename">
          <div class="page-heading">
            <div>
              <p class="eyebrow">Stage 2 of 4</p>
              <h1>Rename PDFs</h1>
              <p class="page-description">Create renamed copies using the letter and 1- to 8-digit number detected on each first page.</p>
            </div>
            <div class="job-actions">
              <button class="button secondary" id="renameAgainButton" type="button">Rename again</button>
              <button class="button" id="renameButton" type="button">Start rename</button>
            </div>
          </div>
          <div class="content-grid">
            <section class="card">
              <div class="card-heading">
                <div><h2>PDF files</h2><p class="muted" style="margin: 5px 0 0; font-size: 11px">Files directly inside the main folder</p></div>
              </div>
              <div id="renameFileList" aria-live="polite"><p class="muted" style="font-size: 12px">No main folder selected yet.</p></div>
            </section>
            <aside class="card">
              <div class="card-heading"><h2>How renaming works</h2></div>
              <p class="muted" style="margin: 0; font-size: 12px; line-height: 1.7">The first page is checked for an identifier. If it cannot be detected, you can enter the ID manually. Existing files are never overwritten.</p>
              <div class="location-card" style="margin-top: 16px"><h3>SAFETY</h3><p>Original PDFs are preserved. Files in subfolders are not included in renaming.</p></div>
            </aside>
          </div>
        </section>
      </main>
    </div>
    <dialog id="completionDialog" class="completion-dialog" aria-labelledby="completionTitle">
      <h2 id="completionTitle">Merge finished</h2>
      <p id="completionMessage"></p>
      <button class="button" id="completionConfirm" type="button">Confirm</button>
    </dialog>

    <script>
      const sourceButton = document.getElementById('sourceButton');
      const pdfButton = document.getElementById('pdfButton');
      const folderInput = document.getElementById('folderInput');
      const pdfInput = document.getElementById('pdfInput');
      const mergeButton = document.getElementById('mergeButton');
      const renameButton = document.getElementById('renameButton');
      const renameAgainButton = document.getElementById('renameAgainButton');
      const clearButton = document.getElementById('clearButton');
      const outputButton = document.getElementById('outputButton');
      const resumeButton = document.getElementById('resumeButton');
      const pauseButton = document.getElementById('pauseButton');
      const stopButton = document.getElementById('stopButton');
      const summary = document.getElementById('summary');
      const folderList = document.getElementById('folderList');
      const sourceLocation = document.getElementById('sourceLocation');
      const outputLocation = document.getElementById('outputLocation');
      const downloadList = document.getElementById('downloadList');
      const renameFileList = document.getElementById('renameFileList');
      const progressPanel = document.getElementById('progressPanel');
      const progressBar = document.getElementById('progressBar');
      const progressTrack = progressPanel.querySelector('[role="progressbar"]');
      const progressText = document.getElementById('progressText');
      const progressPercent = document.getElementById('progressPercent');
      const completionDialog = document.getElementById('completionDialog');
      const completionTitle = document.getElementById('completionTitle');
      const completionMessage = document.getElementById('completionMessage');
      const completionConfirm = document.getElementById('completionConfirm');
      const pageTitle = document.getElementById('pageTitle');
      const dashboardProgress = document.getElementById('dashboardProgress');
      const dashboardJobStatus = document.getElementById('dashboardJobStatus');
      const statsPdfCount = document.getElementById('statsPdfCount');
      const statsFolderCount = document.getElementById('statsFolderCount');
      const statsRenamedCount = document.getElementById('statsRenamedCount');
      const mergeFolderCount = document.getElementById('mergeFolderCount');

      let sourceDirectory = null;
      let sourceRootName = '';
      let sourceFolders = [];
      let sourcePdfGroups = [];
      let activeJob = null;
      let activeJobId = null;
      let lastJobId = null;
      let pauseRequested = false;
      let stopRequested = false;
      let resumePausedJob = null;
      let apiToken = sessionStorage.getItem('pdfWorkshopToken') || '';

      async function apiFetch(url, options = {}) {
        if (!apiToken) {
          apiToken = window.prompt('Enter the PDF Workshop access token:') || '';
          if (apiToken) sessionStorage.setItem('pdfWorkshopToken', apiToken);
        }
        if (!apiToken) throw new Error('An access token is required.');
        const headers = { ...(options.headers || {}), Authorization: `Bearer ${apiToken}` };
        if (options.body && !headers['Content-Type']) headers['Content-Type'] = 'application/json';
        const response = await fetch(url, { ...options, headers });
        const result = await response.json();
        if (!response.ok) {
          if (response.status === 401) {
            sessionStorage.removeItem('pdfWorkshopToken');
            apiToken = '';
          }
          throw new Error(result.error || `Request failed (${response.status}).`);
        }
        return result;
      }

      function showPage(pageName) {
        document.querySelectorAll('[data-page]').forEach(page => {
          page.classList.toggle('is-active', page.dataset.page === pageName);
        });
        document.querySelectorAll('[data-nav]').forEach(button => {
          const isCurrent = button.dataset.nav === pageName;
          if (button.classList.contains('nav-link')) {
            if (isCurrent) button.setAttribute('aria-current', 'page');
            else button.removeAttribute('aria-current');
          }
        });
        pageTitle.textContent = {
          dashboard: 'Dashboard',
          merge: 'Merge PDFs',
          rename: 'Rename PDFs',
        }[pageName] || 'Dashboard';
      }

      document.querySelectorAll('[data-nav]').forEach(button => {
        button.addEventListener('click', () => showPage(button.dataset.nav));
      });

      function updateJobControls() {
        const isActive = activeJob !== null;
        sourceButton.disabled = isActive;
        pdfButton.disabled = isActive;
        outputButton.disabled = isActive;
        mergeButton.disabled = isActive;
        renameButton.disabled = isActive;
        renameAgainButton.disabled = isActive;
        clearButton.disabled = isActive;
        resumeButton.disabled = !isActive || !pauseRequested;
        pauseButton.disabled = !isActive || pauseRequested;
        stopButton.disabled = !isActive || stopRequested;
        if (isActive) {
          dashboardJobStatus.textContent = pauseRequested
            ? `${activeJob === 'rename' ? 'Rename' : 'Merge'} paused`
            : `${activeJob === 'rename' ? 'Renaming' : 'Merging'} PDFs`;
        } else if (progressPanel.hidden) {
          dashboardJobStatus.textContent = 'No job running';
        }
      }

      async function waitWhilePaused() {
        while (pauseRequested && !stopRequested) {
          await new Promise(resolve => {
            resumePausedJob = resolve;
          });
        }
      }

      async function resumeCurrentJob() {
        if (!activeJob || !pauseRequested) return;
        try {
          await apiFetch(`/jobs/${activeJobId}/control`, { method: 'POST', body: JSON.stringify({ action: 'resume' }) });
          pauseRequested = false;
          updateJobControls();
          summary.textContent = `${activeJob === 'rename' ? 'Renaming' : 'Merging'} resumed.`;
        } catch (error) {
          summary.textContent = `Could not resume job: ${error.message}`;
        }
      }

      async function pauseCurrentJob() {
        if (!activeJob || pauseRequested || stopRequested) return;
        try {
          await apiFetch(`/jobs/${activeJobId}/control`, { method: 'POST', body: JSON.stringify({ action: 'pause' }) });
          pauseRequested = true;
          updateJobControls();
          summary.textContent = `${activeJob === 'rename' ? 'Renaming' : 'Merging'} paused after the current ${activeJob === 'rename' ? 'PDF' : 'subfolder'} completes.`;
        } catch (error) {
          summary.textContent = `Could not pause job: ${error.message}`;
        }
      }

      async function stopCurrentJob() {
        if (!activeJob || stopRequested) return;
        try {
          await apiFetch(`/jobs/${activeJobId}/control`, { method: 'POST', body: JSON.stringify({ action: 'stop' }) });
          stopRequested = true;
          pauseRequested = false;
          updateJobControls();
          summary.textContent = `Stopping after the current ${activeJob === 'rename' ? 'PDF' : 'subfolder'} completes...`;
        } catch (error) {
          summary.textContent = `Could not stop job: ${error.message}`;
        }
      }

      function startJob(jobName, jobId) {
        activeJob = jobName;
        activeJobId = jobId;
        pauseRequested = false;
        stopRequested = false;
        updateJobControls();
      }

      function finishJob() {
        activeJob = null;
        activeJobId = null;
        pauseRequested = false;
        stopRequested = false;
        resumePausedJob = null;
        updateJobControls();
      }

      resumeButton.addEventListener('click', resumeCurrentJob);
      pauseButton.addEventListener('click', pauseCurrentJob);
      stopButton.addEventListener('click', stopCurrentJob);

      function updateProgress(completed, total, activeFolder = null, activeFiles = 0) {
        const displayedCompleted = activeFolder ? completed + 1 : completed;
        const percentage = total ? Math.round(displayedCompleted / total * 100) : 0;
        progressPanel.hidden = false;
        progressBar.style.width = `${percentage}%`;
        progressTrack.classList.toggle('is-active', Boolean(activeFolder));
        progressTrack.setAttribute('aria-valuenow', percentage);
        progressPercent.textContent = `${percentage}%`;
        dashboardProgress.textContent = `${percentage}%`;
        progressText.textContent = activeFolder
          ? activeJob === 'rename'
            ? `Renaming ${completed + 1} of ${total}: ${activeFolder}`
            : `Merging ${completed + 1} of ${total}: ${activeFolder} (${activeFiles} PDFs)`
          : activeJob === 'rename'
            ? `${completed} of ${total} PDFs processed`
            : `${completed} of ${total} subfolders processed`;
      }

      function compareFolderNames(left, right) {
        return left.localeCompare(right, undefined, { numeric: true, sensitivity: 'base' });
      }

      function renderFolders(folders) {
        mergeFolderCount.textContent = folders.length
          ? `${folders.length} subfolders · ${folders.reduce((count, folder) => count + folder.pdf_count, 0)} PDFs`
          : 'No PDF subfolders found in this main folder.';
        if (!folders.length) {
          folderList.innerHTML = `<p>${sourceDirectory ? 'No PDF subfolders found in this main folder.' : 'No folder selected yet.'}</p>`;
          return;
        }

        const list = document.createElement('ul');
        folders.forEach(folder => {
          const item = document.createElement('li');
          item.dataset.folderName = folder.name;
          item.textContent = `${folder.name} (${folder.pdf_count} PDFs)`;
          list.append(item);
        });
        folderList.replaceChildren(list);
      }

      function renderPdfFiles(folders, rootName = '', rootPdfFiles = []) {
        const groupsToRender = [
          ...folders,
          ...(rootPdfFiles.length
            ? [{
                name: rootName,
                display_name: `Main folder (${rootName})`,
                pdf_count: rootPdfFiles.length,
                pdf_files: rootPdfFiles,
              }]
            : []),
        ];
        if (!groupsToRender.length) {
          renameFileList.innerHTML = '<p>No PDF files found directly in the selected folder.</p>';
          return;
        }

        const groups = document.createDocumentFragment();
        groupsToRender.forEach(folder => {
          const heading = document.createElement('h3');
          const folderLabel = folder.display_name
            || (folder.is_root ? `Main folder (${folder.name})` : folder.name);
          heading.textContent = `${folderLabel} (${folder.pdf_count} PDFs)`;
          groups.append(heading);

          const list = document.createElement('ul');
          (folder.results || folder.pdf_files.map(filename => ({
            filename,
            status: /^[A-Z]\\s[0-9]{1,8}$/i.test(
              filename.toLocaleLowerCase().endsWith('.pdf') ? filename.slice(0, -4) : filename
            )
              ? 'completed'
              : 'pending',
          }))).forEach(result => {
            const item = document.createElement('li');
            item.classList.add('rename-file');
            updatePdfFileItem(item, result);
            list.append(item);
          });
          groups.append(list);
        });
        renameFileList.replaceChildren(groups);
      }

      function updatePdfFileItem(item, result, onManualResult = null) {
        const statusText = {
          pending: 'waiting',
          renamed: 'completed',
          unchanged: 'completed',
          completed: 'completed',
          skipped: 'skipped',
          failed: 'failed',
        }[result.status] || result.status;
        item.dataset.status = result.status;
        const label = document.createElement('span');
        const displayedName = result.output_filename
          ? `${result.filename} → ${result.output_filename}`
          : result.filename;
        label.textContent = `${displayedName} — ${statusText}${result.message ? `: ${result.message}` : ''}`;
        item.replaceChildren(label);
        item.classList.toggle(
          'is-complete',
          result.status === 'renamed' || result.status === 'unchanged' || result.status === 'completed'
        );
        item.classList.toggle('is-failed', result.status === 'skipped' || result.status === 'failed');
        if (result.status === 'failed') {
          const manualRenameButton = document.createElement('button');
          manualRenameButton.className = 'secondary manual-rename-button';
          manualRenameButton.type = 'button';
          manualRenameButton.textContent = 'Enter ID';
          manualRenameButton.addEventListener('click', async () => {
            const identifier = window.prompt(
              'Enter the letter and 1- to 8-digit ID from this PDF (for example, I 10009):'
            );
            if (identifier === null) return;
            if (!/^[A-Za-z]\\s*\\d{1,8}$/.test(identifier.trim())) {
              summary.textContent = 'Enter one letter followed by a 1- to 8-digit ID.';
              return;
            }

            manualRenameButton.disabled = true;
            try {
              const queued = await apiFetch('/jobs/manual-rename', {
                method: 'POST',
                body: JSON.stringify({
                  workspace_id: sourceDirectory,
                  relative_path: result.relative_path || result.filename,
                  identifier: identifier.trim(),
                }),
              });
              const jobResult = await runQueuedJob(queued.job_id, 'rename');
              const fileResult = jobResult.results?.[0];
              if (!fileResult) throw new Error(`No rename result was returned for ${result.filename}.`);
              lastJobId = queued.job_id;
              updatePdfFileItem(item, fileResult);
              onManualResult?.(fileResult);
              summary.textContent = fileResult.status === 'renamed'
                ? `Renamed ${result.filename} to ${fileResult.filename} using the entered ID.`
                : `${fileResult.filename}: ${fileResult.message || fileResult.status}.`;
            } catch (error) {
              summary.textContent = `Could not apply the entered ID: ${error.message}`;
              manualRenameButton.disabled = false;
            }
          });
          item.append(manualRenameButton);
        }
      }

      function markFolderComplete(folderName) {
        const item = [...folderList.querySelectorAll('[data-folder-name]')]
          .find(entry => entry.dataset.folderName === folderName);
        if (item) {
          item.classList.remove('is-failed');
          item.classList.add('is-complete');
          item.removeAttribute('title');
        }
      }

      function markFolderFailed(folderName, message) {
        const item = [...folderList.querySelectorAll('[data-folder-name]')]
          .find(entry => entry.dataset.folderName === folderName);
        if (item) {
          item.classList.remove('is-complete');
          item.classList.add('is-failed');
          item.title = message;
        }
      }

      function clearDashboard() {
        sourceDirectory = null;
        sourceRootName = '';
        sourceFolders = [];
        sourcePdfGroups = [];
        lastJobId = null;
        sessionStorage.removeItem('pdfWorkshopWorkspace');
        sessionStorage.removeItem('pdfWorkshopJob');
        sourceLocation.textContent = 'Source: No PDFs uploaded.';
        outputLocation.textContent = 'Processed PDFs are stored securely for a limited time.';
        downloadList.replaceChildren();
        renameFileList.innerHTML = '<p>No main folder selected yet.</p>';
        folderList.innerHTML = '<p>No folder selected yet.</p>';
        summary.textContent = '';
        progressPanel.hidden = true;
        progressBar.style.width = '0%';
        progressTrack.classList.remove('is-active');
        progressTrack.setAttribute('aria-valuenow', '0');
        progressText.textContent = 'Ready';
        progressPercent.textContent = '0%';
        dashboardProgress.textContent = '0%';
        dashboardJobStatus.textContent = 'No job running';
        statsPdfCount.textContent = '0';
        statsFolderCount.textContent = '0';
        statsRenamedCount.textContent = '0';
        mergeFolderCount.textContent = 'Choose a main folder to begin.';
        folderInput.value = '';
        pdfInput.value = '';
      }

      async function uploadFiles(selectedFiles, directoryMode) {
        const pdfFiles = [...selectedFiles].filter(file =>
          file.type === 'application/pdf' || file.name.toLowerCase().endsWith('.pdf')
        );
        if (!pdfFiles.length) {
          summary.textContent = 'Select at least one PDF file.';
          return;
        }
        const rootName = directoryMode
          ? (pdfFiles[0].webkitRelativePath.split('/')[0] || 'Uploaded folder')
          : 'Selected PDFs';
        const entries = pdfFiles.map(file => {
          const parts = directoryMode ? file.webkitRelativePath.split('/') : [file.name];
          if (directoryMode) parts.shift();
          return { file, relative_path: parts.join('/'), size: file.size };
        });
        if (entries.some(entry => !entry.relative_path || entry.file.size <= 0)) {
          summary.textContent = 'A selected PDF has an invalid path or is empty.';
          return;
        }
        sourceButton.disabled = true;
        pdfButton.disabled = true;
        summary.textContent = `Preparing ${entries.length} PDF uploads...`;
        try {
          const workspace = await apiFetch('/workspaces', {
            method: 'POST',
            body: JSON.stringify({
              root_name: rootName,
              files: entries.map(({ relative_path, size }) => ({ relative_path, size })),
            }),
          });
          for (let index = 0; index < workspace.uploads.length; index += 1) {
            const upload = workspace.uploads[index];
            const entry = entries.find(candidate => candidate.relative_path === upload.relative_path);
            summary.textContent = `Uploading PDF ${index + 1} of ${entries.length}: ${entry.file.name}`;
            const formData = new FormData();
            Object.entries(upload.fields).forEach(([name, value]) => formData.append(name, value));
            formData.append('file', entry.file);
            const response = await fetch(upload.url, { method: 'POST', body: formData });
            if (!response.ok) throw new Error(`Could not upload ${entry.file.name} (${response.status}).`);
          }
          sourceDirectory = workspace.workspace_id;
          sourceRootName = workspace.root_name;
          sourceFolders = workspace.folders.sort((left, right) => compareFolderNames(left.name, right.name));
          sourcePdfGroups = workspace.root_files.length ? [{
            name: sourceRootName,
            is_root: true,
            pdf_count: workspace.root_files.length,
            pdf_files: workspace.root_files,
          }] : [];
          sessionStorage.setItem('pdfWorkshopWorkspace', JSON.stringify({
            workspace_id: sourceDirectory,
            root_name: sourceRootName,
            folders: sourceFolders,
            root_files: workspace.root_files,
            file_count: entries.length,
          }));
          sourceLocation.textContent = `Source: ${sourceRootName} (${entries.length} PDFs)`;
          renderFolders(sourceFolders);
          renderPdfFiles(sourcePdfGroups);
          statsPdfCount.textContent = String(entries.length);
          statsFolderCount.textContent = String(sourceFolders.length);
          statsRenamedCount.textContent = '0';
          summary.textContent = `${entries.length} PDFs uploaded. Originals are preserved.`;
          updateProgress(0, entries.length);
        } catch (error) {
          summary.textContent = `Could not upload PDFs: ${error.message}`;
        } finally {
          sourceButton.disabled = false;
          pdfButton.disabled = false;
        }
      }

      sourceButton.addEventListener('click', () => folderInput.click());
      pdfButton.addEventListener('click', () => pdfInput.click());
      folderInput.addEventListener('change', () => {
        if (folderInput.files.length) uploadFiles(folderInput.files, true);
      });
      pdfInput.addEventListener('change', () => {
        if (pdfInput.files.length) uploadFiles(pdfInput.files, false);
      });

      async function showDownloads() {
        if (!lastJobId) {
          summary.textContent = 'Complete a rename or merge job before downloading results.';
          return;
        }
        try {
          const result = await apiFetch(`/jobs/${lastJobId}/downloads`);
          downloadList.replaceChildren();
          result.downloads.forEach(download => {
            const link = document.createElement('a');
            link.href = download.url;
            link.textContent = `Download ${download.filename}`;
            link.rel = 'noopener';
            link.target = '_blank';
            downloadList.append(link, document.createElement('br'));
          });
          summary.textContent = result.downloads.length
            ? `${result.downloads.length} secure download link(s) are ready.`
            : 'This job has no downloadable PDF outputs.';
        } catch (error) {
          summary.textContent = `Could not load downloads: ${error.message}`;
        }
      }

      outputButton.addEventListener('click', showDownloads);

      async function renamePdfs() {
        if (!sourceDirectory) {
          summary.textContent = 'Upload a folder or select PDFs first.';
          return;
        }
        const filenames = sourcePdfGroups.flatMap(group => group.pdf_files);
        if (!filenames.length) {
          summary.textContent = 'There are no PDFs directly in the uploaded folder to rename.';
          return;
        }
        if (!window.confirm(`Create renamed copies for ${filenames.length} PDFs directly inside ${sourceRootName}? Original uploads will be preserved.`)) return;
        try {
          const queued = await apiFetch('/jobs', {
            method: 'POST',
            body: JSON.stringify({ workspace_id: sourceDirectory, kind: 'rename' }),
          });
          lastJobId = queued.job_id;
          await runQueuedJob(queued.job_id, 'rename');
        } catch (error) {
          summary.textContent = `Could not rename PDFs: ${error.message}`;
        }
      }

      async function runQueuedJob(jobId, jobName) {
        lastJobId = jobId;
        sessionStorage.setItem('pdfWorkshopJob', JSON.stringify({ job_id: jobId, kind: jobName }));
        startJob(jobName, jobId);
        let job;
        try {
          while (true) {
            job = await apiFetch(`/jobs/${jobId}`);
            pauseRequested = job.status === 'paused';
            stopRequested = job.status === 'stopping' || job.status === 'stopped';
            updateJobControls();
            updateProgress(job.completed || 0, job.total || 0);
            progressText.textContent = job.current
              ? `${jobName === 'rename' ? 'Renaming' : 'Merging'}: ${job.current}`
              : `${job.completed || 0} of ${job.total || 0} ${jobName === 'rename' ? 'PDFs' : 'subfolders'} processed`;
            if (jobName === 'rename' && job.results) {
              renderPdfFiles([{
                name: sourceRootName,
                is_root: true,
                pdf_count: job.results.length,
                results: job.results,
              }]);
              statsRenamedCount.textContent = String(job.results.filter(result => result.status === 'renamed').length);
            }
            if (jobName === 'merge' && job.results) {
              job.results.forEach(result => result.status === 'completed'
                ? markFolderComplete(result.filename)
                : result.status === 'failed' && markFolderFailed(result.filename, result.message));
            }
            if (['completed', 'failed', 'stopped'].includes(job.status)) break;
            await new Promise(resolve => window.setTimeout(resolve, 1200));
          }
        } finally {
          finishJob();
        }
        if (jobName === 'merge') {
          const complete = (job.results || []).filter(result => result.status === 'completed').length;
          const failed = (job.results || []).filter(result => result.status === 'failed').length;
          const message = job.error
            ? `Merge ${job.status}: ${job.error}`
            : job.status === 'stopped'
            ? `Stopped. ${complete} subfolders merged; originals remain unchanged.`
            : `${complete} subfolders merged; ${failed} failed. Original PDFs were preserved.`;
          summary.textContent = message;
          completionTitle.textContent = failed ? 'Merge finished with skipped folders' : 'Merge completed';
          completionMessage.textContent = message;
          if (job.status !== 'stopped') completionDialog.showModal();
        } else {
          const results = job.results || [];
          const renamed = results.filter(result => result.status === 'renamed').length;
          const unchanged = results.filter(result => result.status === 'unchanged').length;
          const skipped = results.filter(result => result.status === 'skipped').length;
          const failed = results.filter(result => result.status === 'failed').length;
          summary.textContent = job.error
            ? `Rename ${job.status}: ${job.error}`
            : `Rename ${job.status}: ${renamed} created, ${unchanged} already named, ${skipped} skipped, ${failed} failed.`;
        }
        return job;
      }

      renameButton.addEventListener('click', renamePdfs);
      renameAgainButton.addEventListener('click', renamePdfs);

      clearButton.addEventListener('click', () => {
        clearDashboard();
        summary.textContent = 'Source folder and merge results cleared.';
        showPage('dashboard');
      });

      completionConfirm.addEventListener('click', () => {
        completionDialog.close();
      });

      mergeButton.addEventListener('click', async () => {
        if (!sourceDirectory || !sourceFolders.length) {
          summary.textContent = 'Upload a folder containing PDF subfolders first.';
          return;
        }
        const confirmed = window.confirm(`Merge PDFs from ${sourceFolders.length} subfolders into separate downloadable files? Originals will be preserved.`);
        if (!confirmed) return;
        try {
          const queued = await apiFetch('/jobs', {
            method: 'POST',
            body: JSON.stringify({ workspace_id: sourceDirectory, kind: 'merge' }),
          });
          lastJobId = queued.job_id;
          await runQueuedJob(queued.job_id, 'merge');
        } catch (error) {
          summary.textContent = `Could not start merge: ${error.message}`;
        }
      });

      async function restoreDashboard() {
        const savedWorkspace = sessionStorage.getItem('pdfWorkshopWorkspace');
        if (!savedWorkspace) return;
        try {
          const cached = JSON.parse(savedWorkspace);
          const workspace = await apiFetch(`/workspaces/${cached.workspace_id}`);
          sourceDirectory = workspace.workspace_id;
          sourceRootName = workspace.root_name;
          sourceFolders = workspace.folders;
          sourcePdfGroups = workspace.root_files.length ? [{
            name: sourceRootName,
            is_root: true,
            pdf_count: workspace.root_files.length,
            pdf_files: workspace.root_files,
          }] : [];
          sourceLocation.textContent = `Source: ${sourceRootName} (${workspace.file_count} PDFs)`;
          statsPdfCount.textContent = String(workspace.file_count);
          statsFolderCount.textContent = String(sourceFolders.length);
          renderFolders(sourceFolders);
          renderPdfFiles(sourcePdfGroups);
          const savedJob = sessionStorage.getItem('pdfWorkshopJob');
          if (savedJob) {
            const job = JSON.parse(savedJob);
            lastJobId = job.job_id;
            await runQueuedJob(job.job_id, job.kind);
          }
        } catch (error) {
          summary.textContent = `Could not restore the cloud workspace: ${error.message}`;
        }
      }

      restoreDashboard();
    </script>
  </body>
</html>
"""


def merge_pdf_files(pdf_files: list[Path]) -> bytes:
    writer = PdfWriter()
    for pdf_file in sorted(pdf_files, key=lambda item: natural_sort_key(item.name)):
      reader = PdfReader(str(pdf_file))
      for page in reader.pages:
        writer.add_page(page)
    output = BytesIO()
    writer.write(output)
    return output.getvalue()


def merge_pdf_files_to_path(pdf_files: list[Path], output_path: Path) -> None:
    writer = PdfWriter()
    for pdf_file in sorted(pdf_files, key=lambda item: natural_sort_key(item.name)):
        reader = PdfReader(str(pdf_file))
        for page in reader.pages:
            writer.add_page(page)
    with output_path.open("wb") as output:
        writer.write(output)


def _dark_components(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    import numpy as np

    visited = np.zeros(mask.shape, dtype=bool)
    components = []
    height, width = mask.shape

    for start_y, start_x in zip(*np.nonzero(mask)):
        if visited[start_y, start_x]:
            continue

        visited[start_y, start_x] = True
        pending = [(int(start_y), int(start_x))]
        left = right = int(start_x)
        top = bottom = int(start_y)
        while pending:
            y, x = pending.pop()
            left = min(left, x)
            right = max(right, x)
            top = min(top, y)
            bottom = max(bottom, y)
            for neighbor_y in range(max(0, y - 1), min(height, y + 2)):
                for neighbor_x in range(max(0, x - 1), min(width, x + 2)):
                    if mask[neighbor_y, neighbor_x] and not visited[neighbor_y, neighbor_x]:
                        visited[neighbor_y, neighbor_x] = True
                        pending.append((neighbor_y, neighbor_x))

        components.append((left, top, right + 1, bottom + 1))

    return components


def _entry_letter_from_image(
    image: np.ndarray,
    ocr_results: list,
    number_box: tuple[float, float, float, float],
    number_left: float,
    engine: RapidOCR,
) -> str:
    import numpy as np

    _, number_top, _, number_bottom = number_box
    number_height = number_bottom - number_top
    number_center = (number_top + number_bottom) / 2

    letter_candidates = []
    for box, text, score in ocr_results:
        match = re.fullmatch(r"\W*([A-Za-z])\W*", text.strip())
        if not match:
            continue

        right = max(point[0] for point in box)
        top = min(point[1] for point in box)
        bottom = max(point[1] for point in box)
        center = (top + bottom) / 2
        distance = number_left - right
        if 0 <= distance <= number_height * 2.5 and abs(center - number_center) <= number_height * 0.6:
            letter_candidates.append((distance, match.group(1).upper(), float(score)))

    if letter_candidates:
        _, letter, _ = min(letter_candidates, key=lambda candidate: (candidate[0], -candidate[2]))
        return letter

    x_start = max(0, int(number_left - number_height * 2.5))
    x_end = min(image.shape[1], int(number_left - 2))
    y_start = max(0, int(number_top - number_height * 0.35))
    y_end = min(image.shape[0], int(number_bottom + number_height * 0.05))
    if x_start >= x_end or y_start >= y_end:
        raise ValueError("No letter immediately before the 1- to 8-digit number was detected.")

    grayscale = image[:, :, :3].astype(np.float32).mean(axis=2)
    components = _dark_components(grayscale[y_start:y_end, x_start:x_end] < 180)
    possible_glyphs = []
    for left, top, right, bottom in components:
        glyph_height = bottom - top
        absolute_left = x_start + left
        absolute_right = x_start + right
        absolute_top = y_start + top
        absolute_bottom = y_start + bottom
        glyph_center = (absolute_top + absolute_bottom) / 2
        if (
            glyph_height >= number_height * 0.5
            and glyph_height <= number_height * 1.4
            and abs(glyph_center - number_center) <= number_height * 0.45
            and number_left - absolute_right <= number_height * 2.5
        ):
            possible_glyphs.append(
                (absolute_left, absolute_top, absolute_right, absolute_bottom)
            )

    for left, top, right, bottom in sorted(
        possible_glyphs,
        key=lambda bounds: bounds[2],
        reverse=True,
    ):
        glyph = (grayscale[top:bottom, left:right] < 180).astype(np.uint8) * 255
        scale = max(2, int(96 / max(glyph.shape)))
        enlarged = np.repeat(np.repeat(glyph, scale, axis=0), scale, axis=1)
        padded = np.full(
            (enlarged.shape[0] + 40, enlarged.shape[1] + 40),
            255,
            dtype=np.uint8,
        )
        padded[20:-20, 20:-20] = enlarged
        glyph_image = np.repeat(padded[:, :, np.newaxis], 3, axis=2)
        recognized, _ = engine.text_rec([glyph_image])
        if recognized:
            recognized_text, confidence = recognized[0]
            character = recognized_text.strip().upper()
            if re.fullmatch(r"[A-Z]", character) and confidence >= 0.25:
                return character
            if (
                character in {"1", "!", "|"}
                and (right - left) / (bottom - top) <= 0.35
            ):
                return "I"

        if (right - left) / (bottom - top) <= 0.35:
            return "I"

    raise ValueError("No letter immediately before the 1- to 8-digit number was detected.")


def _declaration_identifier_from_first_page(pdf_path: Path, engine: RapidOCR) -> str | None:
    import fitz
    import numpy as np

    with fitz.open(pdf_path) as document:
        if not document:
            return None
        page = document[0]
        header_region = fitz.Rect(
            0,
            page.rect.height * 0.03,
            page.rect.width * 0.65,
            page.rect.height * 0.13,
        )
        pixmap = page.get_pixmap(
            matrix=fitz.Matrix(4, 4),
            clip=header_region,
            alpha=False,
        )

    image = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
        pixmap.height, pixmap.width, pixmap.n
    )
    results, _ = engine(image, use_cls=False)
    if not results:
        return None

    declaration_boxes = []
    identifier_candidates = []
    for box, text, confidence in results:
        left = min(point[0] for point in box)
        right = max(point[0] for point in box)
        top = min(point[1] for point in box)
        bottom = max(point[1] for point in box)
        if re.search(r"\bdeclaration\b", text, re.IGNORECASE):
            declaration_boxes.append((left, right, top, bottom))

        match = re.search(r"(?<![A-Za-z0-9])([A-Z])\s*(\d{1,8})(?!\d)", text)
        if match:
            text_width = max(len(text), 1)
            candidate_left = left + (right - left) * match.start() / text_width
            candidate_top = top + (bottom - top) * match.start() / text_width
            candidate_bottom = candidate_top + (bottom - top) * len(match.group()) / text_width
            identifier_candidates.append((
                candidate_left,
                candidate_top,
                candidate_bottom,
                match.group(1).upper(),
                match.group(2),
                float(confidence),
                bool(re.search(r"\bdeclaration\b", text, re.IGNORECASE)),
            ))

    matching_candidates = []
    for candidate in identifier_candidates:
        left, top, bottom, letter, number, confidence, contains_label = candidate
        if contains_label or any(
            left >= label_right
            and abs((top + bottom) / 2 - (label_top + label_bottom) / 2)
            <= max(bottom - top, label_bottom - label_top)
            for _, label_right, label_top, label_bottom in declaration_boxes
        ):
            matching_candidates.append((confidence, letter, number))

    if not matching_candidates:
        return None

    _, letter, number = max(matching_candidates, key=lambda candidate: candidate[0])
    return f"{letter} {number}"


def _report_identifier_from_pdf(pdf_path: Path, engine: RapidOCR) -> str | None:
    import fitz
    import numpy as np

    matching_candidates = []
    report_pattern = re.compile(
      r"\bPNH02\b.{0,40}?\b20\d{2}\b.{0,20}?\b[I1|]\s*(\d{1,8})\b",
      re.IGNORECASE,
    )

    with fitz.open(pdf_path) as document:
      if not document:
        return None
      for page_number in range(min(1, len(document))):
        page = document[page_number]
        report_region = fitz.Rect(
          page.rect.width * 0.25,
          page.rect.height * 0.25,
          page.rect.width * 0.80,
          page.rect.height * 0.42,
        )
        pixmap = page.get_pixmap(
          matrix=fitz.Matrix(4, 4),
          clip=report_region,
          alpha=False,
        )
        image = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
          pixmap.height, pixmap.width, pixmap.n
        )
        results, _ = engine(image, use_cls=False)
        for _, text, confidence in results or []:
          match = report_pattern.search(text)
          if match and float(confidence) >= 0.75:
            matching_candidates.append((
              page_number,
              float(confidence),
              match.group(1),
            ))

    if not matching_candidates:
      return None
    _, _, number = min(
      matching_candidates,
      key=lambda candidate: (candidate[0], -candidate[1]),
    )
    return f"I {number}"


def pdf_identifier_from_first_page(pdf_path: Path) -> str:
    global ocr_engine
    import fitz
    import numpy as np

    with fitz.open(pdf_path) as document:
        if not document:
            raise ValueError("PDF has no pages.")
        first_page_text = document[0].get_text("text")

    report_text_match = re.search(
        r"\bPNH02\b.{0,40}?\b20\d{2}\b.{0,20}?\b[I1|]\s*(\d{1,8})\b",
        first_page_text,
        re.IGNORECASE,
    )
    if report_text_match:
        return f"I {report_text_match.group(1)}"
    entry_text_match = re.search(
        r"\bEntry\s+(?:No\.?|Number)\s*[:#-]?\s*([A-Z])\s*(\d{1,8})\b",
        first_page_text,
        re.IGNORECASE,
    )
    if entry_text_match:
        return f"{entry_text_match.group(1).upper()} {entry_text_match.group(2)}"
    declaration_text_match = re.search(
        r"\bdeclaration\b.{0,80}?\b([A-Z])\s*(\d{1,8})\b",
        first_page_text,
        re.IGNORECASE | re.DOTALL,
    )
    if declaration_text_match:
        return f"{declaration_text_match.group(1).upper()} {declaration_text_match.group(2)}"

    engine = ocr_engine
    if engine is None:
        from rapidocr_onnxruntime import RapidOCR

        engine = RapidOCR(intra_op_num_threads=2)
        ocr_engine = engine

    report_identifier = _report_identifier_from_pdf(pdf_path, engine)
    if report_identifier:
        return report_identifier

    with fitz.open(pdf_path) as document:
        if not document:
            raise ValueError("PDF has no pages.")
        page = document[0]
        scan_regions = [
            (0.08, 0.125, 0.26, 0.19, 5, False),
            (0.06, 0.115, 0.30, 0.205, 6, True),
            (0.035, 0.135, 0.32, 0.22, 8, True),
            (0.03, 0.10, 0.40, 0.24, 6, True),
        ]
        last_letter_error = None
        for left, top, right, bottom, scale, require_entry_label in scan_regions:
            entry_number_region = fitz.Rect(
                page.rect.width * left,
                page.rect.height * top,
                page.rect.width * right,
                page.rect.height * bottom,
            )
            pixmap = page.get_pixmap(
                matrix=fitz.Matrix(scale, scale),
                clip=entry_number_region,
                alpha=False,
            )
            image = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
                pixmap.height, pixmap.width, pixmap.n
            )
            ocr_results, _ = engine(image, use_cls=False)
            if not ocr_results:
                continue

            candidates = []
            entry_labels = []
            for box, text, _ in ocr_results:
                box_left = min(point[0] for point in box)
                box_right = max(point[0] for point in box)
                box_top = min(point[1] for point in box)
                box_bottom = max(point[1] for point in box)
                if re.search(r"\bno\b", text, re.IGNORECASE):
                    entry_labels.append((box_right, box_top, box_bottom))

                text_without_dates = re.sub(
                    r"\b\d{1,2}[/-]\d{1,2}[/-]\d{4}\b", " ", text
                )
                for match in re.finditer(r"(?<!\d)\d{1,8}(?!\d)", text_without_dates):
                    text_width = max(len(text_without_dates), 1)
                    number_left = box_left + (
                        (box_right - box_left) * match.start() / text_width
                    )
                    candidates.append((
                        number_left,
                        match.group(),
                        (box_left, box_top, box_right, box_bottom),
                        None,
                    ))

                if require_entry_label:
                    attached_i_match = re.search(
                        r"[1Il|]\s*(\d{1,8})$",
                        text_without_dates,
                    )
                    if attached_i_match:
                        text_width = max(len(text_without_dates), 1)
                        number_left = box_left + (
                            (box_right - box_left)
                            * attached_i_match.start(1)
                            / text_width
                        )
                        candidates.append((
                            number_left,
                            attached_i_match.group(1),
                            (box_left, box_top, box_right, box_bottom),
                            "I",
                        ))

            if require_entry_label:
                candidates = [
                    candidate for candidate in candidates
                    if any(
                        candidate[0] >= label_right
                        and abs(
                            (candidate[2][1] + candidate[2][3]) / 2
                            - (label_top + label_bottom) / 2
                        ) <= max(
                            candidate[2][3] - candidate[2][1],
                            label_bottom - label_top,
                        )
                        for label_right, label_top, label_bottom in entry_labels
                    )
                ]

            if not candidates:
                continue

            number_left, number, number_box, letter_hint = min(
                candidates, key=lambda candidate: candidate[0]
            )
            if letter_hint:
                return f"{letter_hint} {number}"
            try:
                letter = _entry_letter_from_image(
                    image, ocr_results, number_box, number_left, engine
                )
            except ValueError as error:
                last_letter_error = error
                continue
            return f"{letter} {number}"

    declaration_identifier = _declaration_identifier_from_first_page(pdf_path, engine)
    if declaration_identifier:
        return declaration_identifier
    if last_letter_error:
        raise last_letter_error
    raise ValueError("No 1- to 8-digit number was detected beside the Entry No. label.")


@app.get("/")
def index():
    return render_template_string(HTML_PAGE)


@app.get("/favicon.ico")
def favicon():
  return "", 204


def natural_sort_key(name: str) -> list[object]:
  return [
    (1, int(part)) if part.isdigit() else (0, part.casefold())
    for part in re.split(r"(\d+)", name)
  ]


def _cloud_is_configured() -> bool:
  return all(os.getenv(name) for name in ("AWS_S3_BUCKET", "AWS_DYNAMODB_TABLE", "AWS_SQS_QUEUE_URL"))


@lru_cache(maxsize=3)
def _aws_client(service: str):
  import boto3

  return boto3.client(service, region_name=os.getenv("AWS_REGION", "us-east-1"))


@lru_cache(maxsize=1)
def _dynamodb_table():
  import boto3

  resource = boto3.resource("dynamodb", region_name=os.getenv("AWS_REGION", "us-east-1"))
  return resource.Table(os.environ["AWS_DYNAMODB_TABLE"])


def _require_cloud():
  if not _cloud_is_configured():
    raise RuntimeError("Cloud services are not configured. Set AWS_S3_BUCKET, AWS_DYNAMODB_TABLE, and AWS_SQS_QUEUE_URL.")


def _bucket() -> str:
  _require_cloud()
  return os.environ["AWS_S3_BUCKET"]


@app.before_request
def require_api_token():
  if request.endpoint == "index":
    return None
  expected = os.getenv("PDF_WORKSHOP_ACCESS_TOKEN", "")
  supplied = request.headers.get("Authorization", "")
  if not expected:
    return jsonify({"error": "PDF_WORKSHOP_ACCESS_TOKEN is not configured."}), 503
  if not supplied.startswith("Bearer ") or not hmac.compare_digest(supplied[7:], expected):
    return jsonify({"error": "Authentication required."}), 401
  if not _cloud_is_configured():
    return jsonify({"error": "Cloud storage, job table, and queue are not configured."}), 503
  return None


@app.errorhandler(413)
def request_too_large(_error):
  return jsonify({"error": "Request metadata exceeds the configured limit. PDF bytes must be uploaded directly to cloud storage."}), 413


def _jsonable(value):
  if isinstance(value, Decimal):
    return int(value) if value % 1 == 0 else float(value)
  if isinstance(value, list):
    return [_jsonable(item) for item in value]
  if isinstance(value, dict):
    return {key: _jsonable(item) for key, item in value.items()}
  return value


def _safe_relative_path(value: object) -> str:
  if not isinstance(value, str) or not value:
    raise ValueError("Invalid relative PDF path.")
  try:
    if len(value.encode("utf-8")) > 200:
      raise ValueError("PDF paths must use at most 200 UTF-8 bytes.")
  except UnicodeEncodeError as error:
    raise ValueError("Invalid relative PDF path.") from error
  normalized = value.replace("\\", "/")
  path = PurePosixPath(normalized)
  if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
    raise ValueError("Invalid relative PDF path.")
  if ":" in path.parts[0] or any(ord(char) < 32 for char in normalized):
    raise ValueError("Invalid relative PDF path.")
  if path.suffix.casefold() != ".pdf":
    raise ValueError("Only PDF files can be uploaded.")
  return path.as_posix()


def _workspace(workspace_id: str) -> dict:
  _require_cloud()
  try:
    uuid.UUID(workspace_id)
  except (ValueError, TypeError, AttributeError):
    raise ValueError("Invalid workspace ID.")
  response = _dynamodb_table().get_item(Key={"id": f"workspace:{workspace_id}"})
  item = response.get("Item")
  if not item:
    raise LookupError("Workspace not found.")
  return item


def _group_workspace_files(files: list[dict]) -> tuple[list[dict], list[dict]]:
  root_files = []
  grouped: dict[str, list[str]] = {}
  for item in files:
    relative = PurePosixPath(item["relative_path"])
    if relative.parent == PurePosixPath("."):
      root_files.append(item)
    else:
      grouped.setdefault(relative.parent.as_posix(), []).append(relative.name)
  folders = [
    {"name": name, "pdf_count": len(names), "pdf_files": sorted(names, key=natural_sort_key)}
    for name, names in grouped.items()
  ]
  folders.sort(key=lambda folder: natural_sort_key(folder["name"]))
  root_files.sort(key=lambda item: natural_sort_key(item["relative_path"]))
  return folders, root_files


@app.post("/workspaces")
def create_workspace():
  data = request.get_json(silent=True) or {}
  incoming_files = data.get("files")
  root_name = data.get("root_name", "Uploaded PDFs")
  if not isinstance(incoming_files, list) or not incoming_files or len(incoming_files) > MAX_WORKSPACE_FILES:
    return jsonify({"error": f"Select between 1 and {MAX_WORKSPACE_FILES} PDF files."}), 400
  if not isinstance(root_name, str) or len(root_name) > 160:
    return jsonify({"error": "Invalid folder name."}), 400

  files = []
  seen_paths = set()
  total_bytes = 0
  try:
    for incoming in incoming_files:
      relative_path = _safe_relative_path(incoming.get("relative_path") if isinstance(incoming, dict) else None)
      size = incoming.get("size") if isinstance(incoming, dict) else None
      if not isinstance(size, int) or isinstance(size, bool) or size <= 0 or size > MAX_PDF_BYTES:
        raise ValueError(f"Invalid or oversized PDF: {relative_path}.")
      normalized_key = relative_path.casefold()
      if normalized_key in seen_paths:
        raise ValueError(f"Duplicate file path: {relative_path}.")
      seen_paths.add(normalized_key)
      total_bytes += size
      if total_bytes > MAX_WORKSPACE_BYTES:
        raise ValueError("Selected PDFs exceed the configured workspace size limit.")
      files.append({"relative_path": relative_path, "size": size})
  except (AttributeError, ValueError) as error:
    return jsonify({"error": str(error)}), 400

  workspace_id = str(uuid.uuid4())
  bucket = _bucket()
  s3 = _aws_client("s3")
  uploads = []
  for item in files:
    item["key"] = f"pdf-workshop/{workspace_id}/source/{item['relative_path']}"
    item["filename"] = PurePosixPath(item["relative_path"]).name
    presigned_post = s3.generate_presigned_post(
      Bucket=bucket,
      Key=item["key"],
      Fields={"Content-Type": "application/pdf"},
      Conditions=[
        {"Content-Type": "application/pdf"},
        ["content-length-range", 1, item["size"]],
      ],
      ExpiresIn=3600,
    )
    uploads.append({"relative_path": item["relative_path"], **presigned_post})
  folders, root_files = _group_workspace_files(files)
  item = {
    "id": f"workspace:{workspace_id}",
    "workspace_id": workspace_id,
    "root_name": root_name,
    "files": files,
    "created_at": int(time.time()),
    "expires_at": int(time.time()) + 7 * 24 * 60 * 60,
  }
  _dynamodb_table().put_item(Item=item, ConditionExpression="attribute_not_exists(id)")
  return jsonify({
    "workspace_id": workspace_id,
    "root_name": root_name,
    "uploads": uploads,
    "files": [{"relative_path": file["relative_path"], "filename": file["filename"]} for file in files],
    "folders": folders,
    "root_files": [file["filename"] for file in root_files],
  }), 201


@app.get("/workspaces/<workspace_id>")
def get_workspace(workspace_id: str):
  try:
    workspace = _workspace(workspace_id)
  except ValueError as error:
    return jsonify({"error": str(error)}), 400
  except LookupError as error:
    return jsonify({"error": str(error)}), 404
  folders, root_files = _group_workspace_files(workspace["files"])
  return jsonify({
    "workspace_id": workspace_id,
    "root_name": workspace["root_name"],
    "folders": folders,
    "root_files": [file["filename"] for file in root_files],
    "file_count": len(workspace["files"]),
  })


def _queue_job(workspace_id: str, kind: str, manual_file: str | None = None, identifier: str | None = None) -> dict:
  workspace = _workspace(workspace_id)
  job_id = str(uuid.uuid4())
  table = _dynamodb_table()
  table.update_item(
    Key={"id": f"workspace:{workspace_id}"},
    UpdateExpression="SET active_job = :job, active_job_until = :lease",
    ConditionExpression="attribute_not_exists(active_job) OR active_job_until < :now",
    ExpressionAttributeValues={":job": job_id, ":lease": int(time.time()) + 24 * 60 * 60, ":now": int(time.time())},
  )
  job = {
    "id": f"job:{job_id}",
    "job_id": job_id,
    "workspace_id": workspace_id,
    "kind": kind,
    "status": "queued",
    "created_at": int(time.time()),
    "expires_at": int(time.time()) + 7 * 24 * 60 * 60,
    "completed": 0,
    "total": 0,
    "results": [],
  }
  if manual_file is not None:
    job["manual_file"] = manual_file
    job["identifier"] = identifier
  try:
    table.put_item(Item=job, ConditionExpression="attribute_not_exists(id)")
    _aws_client("sqs").send_message(
      QueueUrl=os.environ["AWS_SQS_QUEUE_URL"],
      MessageBody=json.dumps({"job_id": job_id}),
    )
  except Exception:
    try:
      table.delete_item(Key={"id": f"job:{job_id}"})
    except Exception:
      app.logger.exception("Could not clean up failed job record %s", job_id)
    table.update_item(
      Key={"id": f"workspace:{workspace_id}"},
      UpdateExpression="REMOVE active_job, active_job_until",
      ConditionExpression="active_job = :job",
      ExpressionAttributeValues={":job": job_id},
    )
    raise
  return job


@app.post("/jobs")
def create_job():
  data = request.get_json(silent=True) or {}
  workspace_id = data.get("workspace_id")
  kind = data.get("kind")
  if kind not in {"rename", "merge"}:
    return jsonify({"error": "Job kind must be rename or merge."}), 400
  try:
    job = _queue_job(workspace_id, kind)
  except ValueError as error:
    return jsonify({"error": str(error)}), 400
  except LookupError as error:
    return jsonify({"error": str(error)}), 404
  except Exception as error:
    app.logger.exception("Could not queue %s job", kind)
    error_code = getattr(error, "response", {}).get("Error", {}).get("Code")
    if error_code == "ConditionalCheckFailedException":
      return jsonify({"error": "A job is already active for this workspace."}), 409
    return jsonify({"error": "Could not queue the job. Check cloud service configuration."}), 503
  return jsonify({"job_id": job["job_id"], "status": job["status"]}), 202


@app.post("/jobs/manual-rename")
def create_manual_rename_job():
  data = request.get_json(silent=True) or {}
  workspace_id = data.get("workspace_id")
  try:
    relative_path = _safe_relative_path(data.get("relative_path"))
  except ValueError as error:
    return jsonify({"error": str(error)}), 400
  match = re.fullmatch(r"\s*([A-Za-z])\s*(\d{1,8})\s*", str(data.get("identifier", "")))
  if not match:
    return jsonify({"error": "Enter one letter followed by a 1- to 8-digit ID."}), 400
  try:
    workspace = _workspace(workspace_id)
    if not any(file["relative_path"] == relative_path for file in workspace["files"]):
      return jsonify({"error": "The selected PDF is not in this workspace."}), 404
    job = _queue_job(workspace_id, "manual", relative_path, f"{match.group(1).upper()} {match.group(2)}")
  except ValueError as error:
    return jsonify({"error": str(error)}), 400
  except LookupError as error:
    return jsonify({"error": str(error)}), 404
  except Exception as error:
    app.logger.exception("Could not queue manual rename")
    error_code = getattr(error, "response", {}).get("Error", {}).get("Code")
    if error_code == "ConditionalCheckFailedException":
      return jsonify({"error": "A job is already active for this workspace."}), 409
    return jsonify({"error": "Could not queue manual rename. Check cloud service configuration."}), 503
  return jsonify({"job_id": job["job_id"], "status": job["status"]}), 202


@app.get("/jobs/<job_id>")
def get_job(job_id: str):
  try:
    uuid.UUID(job_id)
  except ValueError:
    return jsonify({"error": "Invalid job ID."}), 400
  _require_cloud()
  response = _dynamodb_table().get_item(Key={"id": f"job:{job_id}"})
  job = response.get("Item")
  if not job:
    return jsonify({"error": "Job not found."}), 404
  return jsonify(_jsonable({key: value for key, value in job.items() if key != "id"}))


@app.post("/jobs/<job_id>/control")
def control_job(job_id: str):
  data = request.get_json(silent=True) or {}
  action = data.get("action")
  target_status = {"pause": "paused", "resume": "running", "stop": "stopping"}.get(action)
  if not target_status:
    return jsonify({"error": "Action must be pause, resume, or stop."}), 400
  response = _dynamodb_table().get_item(Key={"id": f"job:{job_id}"})
  job = response.get("Item")
  if not job:
    return jsonify({"error": "Job not found."}), 404
  if job["status"] in {"completed", "failed", "stopped"}:
    return jsonify({"error": "This job has already finished."}), 409
  _dynamodb_table().update_item(
    Key={"id": f"job:{job_id}"},
    UpdateExpression="SET #status = :status",
    ExpressionAttributeNames={"#status": "status"},
    ExpressionAttributeValues={":status": target_status},
  )
  return jsonify({"status": target_status})


@app.get("/jobs/<job_id>/downloads")
def job_downloads(job_id: str):
  response = _dynamodb_table().get_item(Key={"id": f"job:{job_id}"})
  job = response.get("Item")
  if not job:
    return jsonify({"error": "Job not found."}), 404
  downloads = []
  for result in job.get("results", []):
    if result.get("output_key"):
      downloads.append({
        "filename": result.get("output_filename", result["filename"]),
        "status": result["status"],
        "url": _aws_client("s3").generate_presigned_url(
          "get_object",
          Params={
            "Bucket": _bucket(),
            "Key": result["output_key"],
            "ResponseContentDisposition": "attachment; filename=\"" + re.sub(r"[^A-Za-z0-9._ -]", "_", result.get("output_filename", result["filename"])) + "\"",
          },
          ExpiresIn=600,
        ),
      })
  return jsonify({"downloads": downloads})


def _update_job(job_id: str, **values) -> None:
  names = {"#" + key: key for key in values}
  expression_values = {":" + key: value for key, value in values.items()}
  assignments = ", ".join(f"#{key} = :{key}" for key in values)
  _dynamodb_table().update_item(
    Key={"id": f"job:{job_id}"},
    UpdateExpression=f"SET {assignments}",
    ExpressionAttributeNames=names,
    ExpressionAttributeValues=expression_values,
  )


def _worker_should_continue(job_id: str) -> bool:
  while True:
    response = _dynamodb_table().get_item(Key={"id": f"job:{job_id}"})
    job = response.get("Item")
    if not job or job["status"] in {"stopping", "stopped", "failed", "completed"}:
      return False
    if job["status"] != "paused":
      return True
    time.sleep(2)


def _release_workspace(workspace_id: str, job_id: str) -> None:
  try:
    _dynamodb_table().update_item(
      Key={"id": f"workspace:{workspace_id}"},
      UpdateExpression="REMOVE active_job, active_job_until",
      ConditionExpression="active_job = :job",
      ExpressionAttributeValues={":job": job_id},
    )
  except Exception:
    app.logger.exception("Could not release active-job lock for workspace %s", workspace_id)


def _worker_record(job_id: str, results: list[dict], completed: int, total: int, current: str = "") -> None:
  _update_job(job_id, results=results, completed=completed, total=total, current=current)


def _download_validated_pdf(file: dict, local_path: Path, bucket: str) -> None:
  s3 = _aws_client("s3")
  head = s3.head_object(Bucket=bucket, Key=file["key"])
  if head.get("ContentLength") != file["size"]:
    raise ValueError(f"Upload is incomplete or changed: {file['relative_path']}.")
  s3.download_file(bucket, file["key"], str(local_path))
  with local_path.open("rb") as source_file:
    if source_file.read(5) != b"%PDF-":
      raise ValueError(f"{file['relative_path']} is not a valid PDF.")


def process_job(job_id: str) -> None:
  table = _dynamodb_table()
  job_response = table.get_item(Key={"id": f"job:{job_id}"})
  job = job_response.get("Item")
  if not job or job["status"] in {"completed", "stopped", "failed"}:
    return
  workspace = _workspace(job["workspace_id"])
  files = workspace["files"]
  if job["kind"] in {"rename", "manual"}:
    candidates = [
      item for item in files
      if PurePosixPath(item["relative_path"]).parent == PurePosixPath(".")
      and (job["kind"] != "manual" or item["relative_path"] == job.get("manual_file"))
    ]
  else:
    by_folder: dict[str, list[dict]] = {}
    for item in files:
      parent = PurePosixPath(item["relative_path"]).parent
      if parent != PurePosixPath("."):
        by_folder.setdefault(parent.as_posix(), []).append(item)
    candidates = [
      {"folder": name, "files": sorted(group, key=lambda item: natural_sort_key(item["relative_path"]))}
      for name, group in sorted(by_folder.items(), key=lambda pair: natural_sort_key(pair[0]))
    ]

  total = len(candidates)
  results: list[dict] = []
  initial = table.get_item(Key={"id": f"job:{job_id}"}).get("Item", {})
  if initial.get("status") == "queued":
    try:
      table.update_item(
        Key={"id": f"job:{job_id}"},
        UpdateExpression="SET #status = :running, total = :total",
        ConditionExpression="#status = :queued",
        ExpressionAttributeNames={"#status": "status"},
        ExpressionAttributeValues={":running": "running", ":queued": "queued", ":total": total},
      )
    except Exception:
      pass
  _worker_record(job_id, results, 0, total)
  bucket = _bucket()
  prefix = f"pdf-workshop/{job['workspace_id']}/results/{job_id}"

  try:
    with tempfile.TemporaryDirectory(prefix="pdf-workshop-") as temporary_directory:
      temp_root = Path(temporary_directory)
      for index, candidate in enumerate(candidates):
        if not _worker_should_continue(job_id):
          break
        if job["kind"] in {"rename", "manual"}:
          file = candidate
          filename = PurePosixPath(file["relative_path"]).name
          local_path = temp_root / f"source-{index}.pdf"
          try:
            _download_validated_pdf(file, local_path, bucket)
            identifier = job.get("identifier") if job["kind"] == "manual" else pdf_identifier_from_first_page(local_path)
            if not identifier:
              raise ValueError("No document ID was detected on the first page.")
            target_name = f"{identifier}.pdf"
            existing_names = {
              result.get("output_filename", "").casefold()
              for result in results if result.get("status") in {"renamed", "unchanged"}
            }
            if target_name.casefold() in existing_names:
              result = {"filename": filename, "relative_path": file["relative_path"], "status": "skipped", "message": f"{target_name} is already assigned to another PDF."}
            else:
              output_key = f"{prefix}/renamed/{target_name}"
              _aws_client("s3").upload_file(str(local_path), bucket, output_key, ExtraArgs={"ContentType": "application/pdf"})
              result = {
                "filename": filename,
                "relative_path": file["relative_path"],
                "output_filename": target_name,
                "output_key": output_key,
                "status": "unchanged" if target_name.casefold() == filename.casefold() else "renamed",
                "message": "Original preserved; processed copy is available to download.",
              }
          except Exception as error:
            app.logger.exception("Could not process PDF %s", file["relative_path"])
            result = {"filename": filename, "relative_path": file["relative_path"], "status": "failed", "message": str(error)}
          results.append(result)
          _worker_record(job_id, results, index + 1, total, filename)
        else:
          folder = candidate["folder"]
          local_files = []
          try:
            for file_index, file in enumerate(candidate["files"]):
              local_path = temp_root / f"merge-{index}-{file_index}.pdf"
              _download_validated_pdf(file, local_path, bucket)
              local_files.append(local_path)
            output_path = temp_root / f"merged-{index}.pdf"
            merge_pdf_files_to_path(local_files, output_path)
            output_filename = f"{PurePosixPath(folder).name}.pdf"
            output_key = f"{prefix}/merged/{folder}.pdf"
            _aws_client("s3").upload_file(str(output_path), bucket, output_key, ExtraArgs={"ContentType": "application/pdf"})
            result = {"filename": folder, "output_filename": output_filename, "output_key": output_key, "status": "completed", "message": f"Merged {len(local_files)} PDFs; originals preserved."}
          except Exception as error:
            app.logger.exception("Could not merge PDF folder %s", folder)
            result = {"filename": folder, "status": "failed", "message": str(error)}
          results.append(result)
          _worker_record(job_id, results, index + 1, total, folder)

    final_status = "stopped" if not _worker_should_continue(job_id) else "completed"
    _update_job(job_id, status=final_status, completed=len(results), total=total, current="")
  except Exception as error:
    app.logger.exception("Worker failed for job %s", job_id)
    _update_job(job_id, status="failed", error=str(error))
    raise
  finally:
    _release_workspace(job["workspace_id"], job_id)


def worker_loop() -> None:
  _require_cloud()
  sqs = _aws_client("sqs")
  queue_url = os.environ["AWS_SQS_QUEUE_URL"]
  app.logger.info("PDF Workshop worker polling SQS queue")
  while True:
    response = sqs.receive_message(
      QueueUrl=queue_url,
      MaxNumberOfMessages=1,
      WaitTimeSeconds=20,
            VisibilityTimeout=int(os.getenv("AWS_SQS_VISIBILITY_TIMEOUT", "43200")),
    )
    for message in response.get("Messages", []):
      try:
        process_job(json.loads(message["Body"])["job_id"])
        sqs.delete_message(QueueUrl=queue_url, ReceiptHandle=message["ReceiptHandle"])
      except Exception:
        app.logger.exception("Queued job failed and will be retried by SQS")


if __name__ == "__main__":
  import sys

  if len(sys.argv) > 1 and sys.argv[1] == "worker":
    worker_loop()
  else:
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")))
