from __future__ import annotations

from io import BytesIO
from pathlib import Path
import re
import shutil

import fitz
import numpy as np
from flask import Flask, jsonify, render_template_string, request
from pypdf import PdfReader, PdfWriter
from rapidocr_onnxruntime import RapidOCR

app = Flask(__name__)
source_directory: Path | None = None
output_directory: Path | None = None
ocr_engine: RapidOCR | None = None

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
      .workflow-card {
        margin-bottom: 20px;
      }
      .workflow-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 12px;
      }
      .workflow-step {
        min-height: 100px;
        border: 1px solid #fff;
        border-radius: 18px;
        padding: 15px;
        background: #f7f8fc;
      }
      .workflow-step.current {
        border-color: #171923;
        background: #171923;
        color: #fff;
      }
      .step-label {
        color: #858b9a;
        font-size: 10px;
      }
      .workflow-step.current .step-label,
      .workflow-step.current .step-status {
        color: rgba(255, 255, 255, 0.65);
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
          <strong>Files stay on your device</strong>
          Choose folders from this computer. PDF operations run locally.
        </div>
      </aside>

      <main class="main-content">
        <header class="topbar">
          <div class="breadcrumbs"><span>PDF Workshop</span><span>/</span><strong id="pageTitle">Dashboard</strong></div>
          <div class="top-actions">
            <span class="local-badge">Local workspace</span>
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
              <p class="page-description">A clear view of your PDF workflow. Choose a source folder, then merge subfolders or rename PDFs in the main folder.</p>
            </div>
          </div>

          <section class="card workflow-card">
            <div class="card-heading"><h2>Workflow</h2><span class="muted" style="font-size: 11px">Four simple steps</span></div>
            <div class="workflow-grid">
              <div class="workflow-step">
                <div class="step-label">STEP 1</div><div class="step-name">Choose folder</div><div class="step-status">Select a main folder</div>
              </div>
              <div class="workflow-step">
                <div class="step-label">STEP 2</div><div class="step-name">Rename</div><div class="step-status">Optional · main-folder PDFs</div>
              </div>
              <div class="workflow-step current">
                <div class="step-label">STEP 3</div><div class="step-name">Merge</div><div class="step-status">Process PDF subfolders</div>
              </div>
              <div class="workflow-step">
                <div class="step-label">STEP 4</div><div class="step-name">Save</div><div class="step-status">Choose a destination</div>
              </div>
            </div>
          </section>

          <div class="dashboard-grid">
            <section class="card">
              <div class="card-heading"><h2>Quick actions</h2></div>
              <div class="quick-grid">
                <button class="action-tile" id="sourceButton" type="button">
                  <span class="tile-icon" aria-hidden="true">▣</span><span><strong>Choose main folder</strong><small>Load folders and PDFs</small></span>
                </button>
                <button class="action-tile" id="outputButton" type="button">
                  <span class="tile-icon" aria-hidden="true">⇩</span><span><strong>Choose save destination</strong><small>Select where merged PDFs go</small></span>
                </button>
                <button class="action-tile" type="button" data-nav="merge">
                  <span class="tile-icon" aria-hidden="true">▤</span><span><strong>Merge PDFs</strong><small>Combine files in each subfolder</small></span>
                </button>
                <button class="action-tile" type="button" data-nav="rename">
                  <span class="tile-icon" aria-hidden="true">✎</span><span><strong>Rename PDFs</strong><small>Use the ID on each first page</small></span>
                </button>
              </div>
              <div class="location-grid">
                <div class="location-card"><h3>MAIN FOLDER</h3><p id="sourceLocation">Source folder: Not selected.</p></div>
                <div class="location-card"><h3>SAVE DESTINATION</h3><p id="outputLocation" aria-live="polite">Save location: Not selected. Choose Save Destination to open the Windows folder picker.</p></div>
              </div>
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

          <div class="stat-grid">
            <section class="card stat-card"><p>PDFs in main folder</p><strong id="statsPdfCount">0</strong></section>
            <section class="card stat-card"><p>Subfolders ready to merge</p><strong id="statsFolderCount">0</strong></section>
            <section class="card stat-card"><p>PDFs renamed this run</p><strong id="statsRenamedCount">0</strong></section>
          </div>

          <div class="warning">
            <span aria-hidden="true">⚠</span>
            <div><strong>Permanent delete notice:</strong> Each source subfolder is permanently deleted after its merged PDF is saved. The save destination must be outside the source subfolder.</div>
          </div>
        </section>

        <section class="page" data-page="merge">
          <div class="page-heading">
            <div>
              <p class="eyebrow">Stage 3 of 4</p>
              <h1>Merge PDFs</h1>
              <p class="page-description">Combine the PDFs in each subfolder into its own file. Successfully merged source folders are permanently deleted.</p>
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
              <p class="muted" style="margin: 0; font-size: 12px; line-height: 1.7">Choose both a main folder and save destination from the Dashboard. Each eligible subfolder is saved as a separate merged PDF.</p>
              <div class="warning" style="margin-top: 16px"><span aria-hidden="true">⚠</span><div>Once a merged PDF is safely saved, its source subfolder and contents are permanently removed.</div></div>
            </aside>
          </div>
        </section>

        <section class="page" data-page="rename">
          <div class="page-heading">
            <div>
              <p class="eyebrow">Stage 2 of 4</p>
              <h1>Rename PDFs</h1>
              <p class="page-description">Rename PDFs directly in the selected main folder using the letter and 1- to 8-digit number found on the first page. Existing names are checked against PDF content too.</p>
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
              <div class="location-card" style="margin-top: 16px"><h3>SAFETY</h3><p>Subfolders and the save destination are not included in renaming.</p></div>
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
      let sourceFolders = [];
      let sourcePdfGroups = [];
      let outputDirectory = null;
      let activeJob = null;
      let pauseRequested = false;
      let stopRequested = false;
      let resumePausedJob = null;

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

      function resumeCurrentJob() {
        if (!activeJob || !pauseRequested) return;
        pauseRequested = false;
        resumePausedJob?.();
        resumePausedJob = null;
        updateJobControls();
        summary.textContent = `${activeJob === 'rename' ? 'Renaming' : 'Merging'} resumed.`;
      }

      function pauseCurrentJob() {
        if (!activeJob || pauseRequested || stopRequested) return;
        pauseRequested = true;
        updateJobControls();
        summary.textContent = `${activeJob === 'rename' ? 'Renaming' : 'Merging'} paused after the current ${activeJob === 'rename' ? 'PDF' : 'subfolder'} completes.`;
      }

      function stopCurrentJob() {
        if (!activeJob || stopRequested) return;
        stopRequested = true;
        pauseRequested = false;
        resumePausedJob?.();
        resumePausedJob = null;
        updateJobControls();
        summary.textContent = `Stopping after the current ${activeJob === 'rename' ? 'PDF' : 'subfolder'} completes...`;
      }

      function startJob(jobName) {
        activeJob = jobName;
        pauseRequested = false;
        stopRequested = false;
        updateJobControls();
      }

      function finishJob() {
        activeJob = null;
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
        label.textContent = `${result.filename} — ${statusText}${result.message ? `: ${result.message}` : ''}`;
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
              const response = await fetch('/rename-pdfs', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ filename: result.filename, identifier: identifier.trim() }),
              });
              const responseResult = await response.json();
              if (!response.ok) {
                throw new Error(responseResult.error || `Could not rename ${result.filename}.`);
              }
              const fileResult = responseResult.folders.flatMap(folder => folder.results)[0];
              if (!fileResult) throw new Error(`No rename result was returned for ${result.filename}.`);
              updatePdfFileItem(item, fileResult, onManualResult);
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
        sourceFolders = [];
        sourcePdfGroups = [];
        outputDirectory = null;
        sourceLocation.textContent = 'Source folder: Not selected.';
        outputLocation.textContent = 'Save location: Not selected. Choose Save Destination to open the Windows folder picker.';
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
      }

      sourceButton.addEventListener('click', async () => {
        try {
          summary.textContent = 'Select the main folder that contains the PDF subfolders.';
          const response = await fetch('/choose-source', { method: 'POST' });
          const result = await response.json();
          if (!response.ok) throw new Error(result.error || 'Could not choose the main folder.');
          if (result.cancelled) {
            summary.textContent = 'Main folder selection canceled.';
            return;
          }
          sourceDirectory = result.path;
          sourceFolders = result.folders.sort((left, right) => compareFolderNames(left.name, right.name));
          sourcePdfGroups = result.pdf_groups;
          sourceLocation.textContent = `Source folder: ${sourceDirectory}`;
          renderFolders(sourceFolders);
          renderPdfFiles(sourcePdfGroups);
          const pdfCount = sourcePdfGroups.reduce((count, group) => count + group.pdf_count, 0);
          statsPdfCount.textContent = String(pdfCount);
          statsFolderCount.textContent = String(sourceFolders.length);
          summary.textContent = `${pdfCount} PDFs found directly in the selected folder.`;
          updateProgress(0, sourceFolders.length);
        } catch (error) {
          summary.textContent = `Could not choose main folder: ${error.message}`;
        }
      });

      outputButton.addEventListener('click', async () => {
        try {
          summary.textContent = 'Select the save destination in the Windows folder picker.';
          const response = await fetch('/choose-output', { method: 'POST' });
          const result = await response.json();
          if (!response.ok) throw new Error(result.error || 'Could not choose save destination.');
          if (result.cancelled) {
            summary.textContent = 'Folder selection canceled.';
            return;
          }
          outputDirectory = result.path;
          outputLocation.textContent = `Save location: ${outputDirectory}`;
          summary.textContent = 'Save destination selected.';
        } catch (error) {
          summary.textContent = `Could not choose save destination: ${error.message}`;
        }
      });

      async function renamePdfs() {
        if (!sourceDirectory) {
          summary.textContent = 'Choose a main folder first.';
          return;
        }

        const confirmed = window.confirm(
          `Check and rename PDF files directly inside ${sourceDirectory} using the letter and number from each first page? Incorrect existing names will be corrected. Subfolders and the save destination will not be changed. Existing files will not be overwritten.`
        );
        if (!confirmed) return;

        const filenames = sourcePdfGroups.flatMap(group => group.pdf_files);
        if (!filenames.length) {
          summary.textContent = 'No PDF files were found directly in the selected folder.';
          return;
        }

        const folderName = sourceDirectory.split(/[\\/]/).pop();
        const results = filenames.map(filename => ({ filename, status: 'pending' }));
        startJob('rename');
        updateProgress(0, filenames.length, filenames[0]);
        renderPdfFiles([{
          name: folderName,
          is_root: true,
          pdf_count: results.length,
          results,
        }]);
        summary.textContent = `Reading PDFs directly in ${sourceDirectory}...`;
        try {
          for (let index = 0; index < filenames.length; index += 1) {
            await waitWhilePaused();
            if (stopRequested) break;

            const filename = filenames[index];
            updateProgress(index, filenames.length, filename);
            summary.textContent = `Renaming ${index + 1} of ${filenames.length}: ${filename}`;
            const response = await fetch('/rename-pdfs', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ filename }),
            });
            const result = await response.json();
            if (!response.ok) throw new Error(result.error || `Could not rename ${filename}.`);

            const fileResult = result.folders.flatMap(folder => folder.results)[0];
            if (!fileResult) throw new Error(`No rename result was returned for ${filename}.`);
            results[index] = fileResult;
            const fileItems = renameFileList.querySelectorAll('.rename-file');
            updatePdfFileItem(fileItems[index], fileResult, manualResult => {
              results[index] = manualResult;
              statsRenamedCount.textContent = String(
                results.filter(item => item.status === 'renamed').length
              );
              sourcePdfGroups = sourcePdfGroups.map(group => group.is_root
                ? { ...group, pdf_files: results.map(item => item.filename) }
                : group);
            });
            updateProgress(index + 1, filenames.length);
          }

          const statuses = results.filter(item => item.status !== 'pending');
          const renamed = statuses.filter(item => item.status === 'renamed').length;
          statsRenamedCount.textContent = String(renamed);
          const alreadyNamed = statuses.filter(item => item.status === 'unchanged').length;
          const skipped = statuses.filter(item => item.status === 'skipped').length;
          const failed = statuses.filter(item => item.status === 'failed').length;
          summary.textContent = stopRequested
            ? `Stopped. ${renamed} renamed, ${alreadyNamed} already named, ${skipped} skipped, ${failed} failed; remaining PDFs were not processed.`
            : `Rename finished: ${renamed} renamed, ${alreadyNamed} already named, ${skipped} skipped, ${failed} failed. See each file below for details.`;
        } catch (error) {
          summary.textContent = `Could not rename PDFs: ${error.message}`;
        } finally {
          statsRenamedCount.textContent = String(
            results.filter(item => item.status === 'renamed').length
          );
          sourcePdfGroups = sourcePdfGroups.map(group => group.is_root
            ? { ...group, pdf_files: results.map(item => item.filename) }
            : group);
          finishJob();
          dashboardJobStatus.textContent = 'Rename run finished';
        }
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
        clearDashboard();
      });

      mergeButton.addEventListener('click', async () => {
        if (!sourceDirectory || !sourceFolders.length) {
          summary.textContent = 'Please choose a folder first.';
          return;
        }

        if (!outputDirectory) {
          summary.textContent = 'Choose an output folder first.';
          return;
        }

        const confirmed = window.confirm(
          `Merge ${sourceFolders.length} subfolders into separate PDFs in the save destination, then permanently delete each source subfolder and all its contents? This cannot be undone.`
        );
        if (!confirmed) return;

        let processed = 0;
        let completed = 0;
        const failedFolders = [];
        startJob('merge');
        updateProgress(0, sourceFolders.length);

        try {
          for (const folder of sourceFolders) {
            await waitWhilePaused();
            if (stopRequested) break;

            const folderName = folder.name;
            updateProgress(processed, sourceFolders.length, folderName, folder.pdf_count);
            summary.textContent = `Merging PDFs in ${folderName}...`;
            try {
              const response = await fetch('/merge', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ folder_name: folderName }),
              });
              const result = await response.json();
              if (!response.ok) throw new Error(result.error || `Could not merge ${folderName}`);

              markFolderComplete(folderName);
              completed += 1;
              summary.textContent = `Saved ${completed} PDFs: ${result.filename}; deleted source folder ${folderName}.`;
            } catch (error) {
              failedFolders.push(folderName);
              markFolderFailed(folderName, error.message);
              summary.textContent = `Skipped ${folderName}: ${error.message}. Continuing with the next subfolder...`;
            }

            processed += 1;
            updateProgress(processed, sourceFolders.length);
          }

          const finalSummary = stopRequested
            ? `Stopped. ${completed} subfolders were merged and deleted; ${sourceFolders.length - processed} remain unprocessed.`
            : failedFolders.length
              ? `Finished: ${completed} merged, ${failedFolders.length} skipped (${failedFolders.join(', ')}). PDF errors leave their source folders untouched; review the red entries for details.`
              : `Done. Saved ${completed} PDFs and removed their source subfolders.`;
          summary.textContent = finalSummary;
          dashboardJobStatus.textContent = stopRequested ? 'Merge stopped' : 'Merge run finished';
          if (!stopRequested) {
            completionTitle.textContent = failedFolders.length
              ? 'Merge finished with skipped folders'
              : 'Merge completed';
            completionMessage.textContent = finalSummary;
            completionDialog.showModal();
          }
        } finally {
          finishJob();
        }
      });
    </script>
  </body>
</html>
"""


def merge_pdf_files(pdf_files: list[Path]) -> bytes:
    writer = PdfWriter()
    for pdf_file in sorted(pdf_files, key=lambda item: item.name.lower()):
        reader = PdfReader(str(pdf_file))
        for page in reader.pages:
            writer.add_page(page)
    output = BytesIO()
    writer.write(output)
    return output.getvalue()


def _dark_components(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
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
    matching_candidates = []
    report_pattern = re.compile(
        r"\bPNH02\b.{0,40}?\b20\d{2}\b.{0,20}?\b[I1|]\s*(\d{1,8})\b",
        re.IGNORECASE,
    )

    with fitz.open(pdf_path) as document:
        if not document:
            return None
        for page_number, page in enumerate(document):
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

    engine = ocr_engine
    if engine is None:
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


def natural_sort_key(name: str) -> list[object]:
  return [int(part) if part.isdigit() else part.casefold() for part in re.split(r"(\d+)", name)]


def folders_with_pdfs(directory: Path) -> list[dict[str, str | int | list[str]]]:
  folders = []
  for folder in directory.iterdir():
    if not folder.is_dir():
      continue
    pdf_files = sorted(
      (
        item.name for item in folder.iterdir()
        if item.is_file() and item.suffix.lower() == ".pdf"
      ),
      key=str.casefold,
    )
    if pdf_files:
      folders.append({
        "name": folder.name,
        "pdf_count": len(pdf_files),
        "pdf_files": pdf_files,
      })
  return sorted(folders, key=lambda folder: natural_sort_key(str(folder["name"])))


def pdf_files_in_directory(directory: Path) -> list[Path]:
  return sorted(
    (
      item for item in directory.iterdir()
      if item.is_file() and item.suffix.lower() == ".pdf"
    ),
    key=lambda item: item.name.casefold(),
  )


@app.post("/choose-source")
def choose_source():
  global source_directory

  import tkinter as tk
  from tkinter import filedialog

  root = None
  try:
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    selected_path = filedialog.askdirectory(
      parent=root,
      title="Choose the main folder containing PDF subfolders",
      mustexist=True,
    )
  except Exception:
    app.logger.exception("Could not open the Windows source folder picker")
    return jsonify({"error": "Could not open the Windows folder picker."}), 500
  finally:
    if root is not None:
      root.destroy()

  if not selected_path:
    return jsonify({"cancelled": True})

  source_directory = Path(selected_path)
  pdf_files = pdf_files_in_directory(source_directory)
  return jsonify({
    "path": str(source_directory),
    "pdf_groups": [{
      "name": source_directory.name,
      "is_root": True,
      "pdf_count": len(pdf_files),
      "pdf_files": [item.name for item in pdf_files],
    }] if pdf_files else [],
    "folders": folders_with_pdfs(source_directory),
  })


@app.post("/choose-output")
def choose_output():
    global output_directory

    import tkinter as tk
    from tkinter import filedialog

    root = None
    try:
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        selected_path = filedialog.askdirectory(
            parent=root,
            title="Choose where merged PDFs will be saved",
            mustexist=True,
        )
    except Exception:
        app.logger.exception("Could not open the Windows folder picker")
        return jsonify({"error": "Could not open the Windows folder picker."}), 500
    finally:
        if root is not None:
            root.destroy()

    if not selected_path:
        return jsonify({"cancelled": True})

    output_directory = Path(selected_path)
    return jsonify({"path": str(output_directory)})


@app.post("/rename-pdfs")
def rename_pdfs():
    if source_directory is None or not source_directory.is_dir():
        return jsonify({"error": "Choose an existing main folder first."}), 400

    requested_filename = request.json.get("filename") if request.is_json else None
    requested_identifier = request.json.get("identifier") if request.is_json else None
    if request.is_json and "filename" in request.json:
        if (
            not isinstance(requested_filename, str)
            or Path(requested_filename).name != requested_filename
            or "/" in requested_filename
            or "\\" in requested_filename
            or Path(requested_filename).suffix.lower() != ".pdf"
        ):
            return jsonify({"error": "Invalid PDF filename."}), 400
    if request.is_json and "identifier" in request.json:
        if not isinstance(requested_identifier, str):
            return jsonify({"error": "Invalid PDF identifier. Use one letter and a 1- to 8-digit ID."}), 400
        identifier_match = re.fullmatch(
            r"\s*([A-Za-z])\s*(\d{1,8})\s*",
            requested_identifier,
        )
        if not identifier_match:
            return jsonify({"error": "Invalid PDF identifier. Use one letter and a 1- to 8-digit ID."}), 400
        requested_identifier = f"{identifier_match.group(1).upper()} {identifier_match.group(2)}"

    try:
        if requested_filename is None:
            pdf_files = pdf_files_in_directory(source_directory)
        else:
            selected_pdf = source_directory / requested_filename
            if (
                not selected_pdf.is_file()
                or selected_pdf.resolve().parent != source_directory.resolve()
            ):
                return jsonify({"error": "The selected PDF was not found in the main folder."}), 404
            pdf_files = [selected_pdf]
    except OSError as error:
        app.logger.exception("Could not list PDFs in %s", source_directory)
        return jsonify({"error": f"Could not read the selected folder: {error}"}), 500

    results = []
    for pdf_file in pdf_files:
        try:
            identifier = requested_identifier or pdf_identifier_from_first_page(pdf_file)
            target_path = pdf_file.with_name(f"{identifier}.pdf")
            if target_path.name.casefold() == pdf_file.name.casefold():
                results.append({
                    "filename": pdf_file.name,
                    "status": "unchanged",
                    "message": "Filename matches the ID on the first page",
                })
            elif target_path.exists():
                results.append({
                    "filename": pdf_file.name,
                    "status": "skipped",
                    "message": f"{target_path.name} already exists",
                })
            else:
                pdf_file.rename(target_path)
                results.append({
                    "filename": target_path.name,
                    "status": "renamed",
                    "message": f"from {pdf_file.name}",
                })
        except ValueError as error:
            app.logger.warning("Could not identify PDF filename for %s: %s", pdf_file, error)
            results.append({
                "filename": pdf_file.name,
                "status": "failed",
                "message": str(error),
            })
        except Exception as error:
            app.logger.exception("Could not rename PDF %s", pdf_file)
            results.append({
                "filename": pdf_file.name,
                "status": "failed",
                "message": str(error),
            })

    total = len(results)
    return jsonify({
        "total": total,
        "renamed": sum(result["status"] == "renamed" for result in results),
        "folders": [{
            "name": source_directory.name,
            "is_root": True,
            "pdf_count": total,
            "results": results,
        }] if results else [],
    })


@app.post("/merge")
def merge():
    if source_directory is None or not source_directory.is_dir():
        return jsonify({"error": "Choose an existing main folder first."}), 400

    if output_directory is None or not output_directory.is_dir():
        return jsonify({"error": "Choose an existing save destination first."}), 400

    folder_name = request.json.get("folder_name", "") if request.is_json else ""
    if not folder_name or Path(folder_name).name != folder_name or "/" in folder_name or "\\" in folder_name:
        return jsonify({"error": "Invalid subdirectory name."}), 400

    source_folder = source_directory / folder_name
    if not source_folder.is_dir() or source_folder.resolve().parent != source_directory.resolve():
        return jsonify({"error": "The selected source subfolder was not found."}), 404
    if source_folder.resolve() in output_directory.resolve().parents or output_directory.resolve() == source_folder.resolve():
      return jsonify({"error": "The save destination cannot be inside the source subfolder being deleted."}), 400

    pdf_files = [
        item for item in source_folder.iterdir()
        if item.is_file() and item.suffix.lower() == ".pdf"
    ]
    if not pdf_files:
        return jsonify({"error": "No PDFs remain in this source subfolder."}), 400

    try:
        merged_pdf = merge_pdf_files(pdf_files)
    except Exception as error:
      app.logger.exception("Could not read or merge PDFs in %s", folder_name)
      return jsonify({"error": f"PDF merge failed in {folder_name}; source folder was left untouched: {error}"}), 422

    try:
        output_path = output_directory / f"{folder_name}.pdf"
        output_path.write_bytes(merged_pdf)
    except Exception as error:
      app.logger.exception("Could not save merged PDF for %s", folder_name)
      return jsonify({"error": f"Could not save {folder_name}.pdf; source folder was left untouched: {error}"}), 500

    try:
        shutil.rmtree(source_folder)
    except Exception as error:
      app.logger.exception("Could not remove source folder %s", source_folder)
      return jsonify({"error": f"Saved {folder_name}.pdf, but source folder cleanup failed: {error}"}), 500

    return jsonify({"filename": output_path.name, "path": str(output_path), "source_removed": True})


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
