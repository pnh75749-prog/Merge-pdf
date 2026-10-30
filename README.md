# PDF Workshop

Flask dashboard for browser-based PDF uploads, OCR-based copy renaming, and per-subfolder merging. The Vercel function handles the dashboard and short cloud API calls. PDF processing runs in a separate SQS worker so OCR and merges do not depend on Vercel execution time or temporary storage.

## Architecture

- Browser uploads PDF bytes directly to private Amazon S3 using short-lived, size-constrained presigned POST policies. PDF bytes never pass through a Vercel request body.
- DynamoDB stores workspace metadata, job state, progress, and per-file results.
- SQS queues rename and merge work. Run `python dashboard_app.py worker` on an always-on worker host with access to the same AWS resources.
- The worker downloads PDFs to temporary storage, processes them, uploads separate output copies, and leaves original uploads unchanged.
- Download URLs are private S3 presigned URLs that expire after 10 minutes. Configure an S3 lifecycle rule to expire the `pdf-workshop/` prefix after 7 days.

## AWS Setup

1. Create a private S3 bucket. Configure bucket CORS to allow `POST` from the Vercel deployment origin and local origin, allow the `Content-Type` request header, and expose no credentials. Keep block-public-access enabled.
2. Create a DynamoDB table with a string partition key named `id`. Enable TTL on the numeric `expires_at` attribute. Workspace metadata expires after seven days; job records should also be covered by a table TTL policy or a scheduled cleanup.
3. Create an SQS standard queue and a dead-letter queue. Set the source queue visibility timeout to 12 hours and configure a small `maxReceiveCount` for the dead-letter queue. The worker's default receive visibility timeout is 12 hours.
4. Create an IAM identity for Vercel with only the required bucket object, DynamoDB table, and SQS send permissions. Give the worker its own IAM role with S3 object read/write, DynamoDB read/write, and SQS receive/delete permissions. Prefer role-based credentials on the worker host; never commit AWS secrets.
5. Set the AWS service variables below in Vercel and on the worker. Use the same region, bucket, table, and queue on both. Set `PDF_WORKSHOP_ACCESS_TOKEN` on Vercel and the local web process only; make it a long random secret. Dashboard users enter it into the browser prompt and it remains only in that tab's session storage.
6. Start the worker before users submit jobs. A Vercel deployment alone cannot consume SQS or run a persistent worker.

### Required environment variables

- `AWS_REGION`
- `AWS_S3_BUCKET`
- `AWS_DYNAMODB_TABLE`
- `AWS_SQS_QUEUE_URL`
- `PDF_WORKSHOP_ACCESS_TOKEN`
- On Vercel, also set `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` for the restricted Vercel IAM identity, plus `AWS_SESSION_TOKEN` if using temporary credentials. On the worker, use the platform's IAM role when available.

Optional limits: `MAX_PDF_BYTES` defaults to 500 MiB per PDF, `MAX_WORKSPACE_BYTES` defaults to 2 GiB per selection, paths are limited to 200 UTF-8 bytes, and the workspace accepts up to 100 PDF files. Ensure worker disk and memory can accommodate the selected files; OCR should use a worker with several GiB of RAM.

## Local Development (PowerShell)

From the repository root:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
$env:AWS_REGION = "us-east-1"
$env:AWS_S3_BUCKET = "your-private-bucket"
$env:AWS_DYNAMODB_TABLE = "pdf-workshop-jobs"
$env:AWS_SQS_QUEUE_URL = "https://sqs.us-east-1.amazonaws.com/123456789012/pdf-workshop"
python dashboard_app.py worker
```

Run the Flask dashboard in a second terminal with the AWS service variables and access token:

```powershell
.\.venv\Scripts\Activate.ps1
$env:AWS_REGION = "us-east-1"
$env:AWS_S3_BUCKET = "your-private-bucket"
$env:AWS_DYNAMODB_TABLE = "pdf-workshop-jobs"
$env:AWS_SQS_QUEUE_URL = "https://sqs.us-east-1.amazonaws.com/123456789012/pdf-workshop"
$env:PDF_WORKSHOP_ACCESS_TOKEN = "set-a-long-random-secret"
python -m flask --app app run --debug
```

Open `http://127.0.0.1:5000`. These local commands require AWS credentials available through the normal AWS SDK credential chain. Do not use the example values above as production credentials.

## Vercel Deployment

Install Node.js LTS (which includes npm) if it is not already installed. Then install and authenticate the Vercel CLI and link the project:

```powershell
npm install --global vercel
vercel login
vercel link
```

Add the required environment variables in the Vercel project dashboard under **Settings → Environment Variables** for Preview and Production. For command-line setup, run `vercel env add VARIABLE_NAME` once for each required variable and enter the value when prompted. Pull the configured values for local Vercel testing:

```powershell
vercel env pull .env.local
```

Ensure the separate worker is running, then develop and deploy:

```powershell
vercel dev
vercel --prod
```

Vercel auto-detects the root `app.py` entrypoint and its `app` variable. The root `.python-version` pins the runtime to 3.12. `vercel.json` only excludes local-only files from the Python function bundle; it does not override Vercel's Python framework routing.

## Worker Deployment

Deploy this repository to an always-on container or VM with the AWS worker role and the same cloud variables. Install with `pip install -r requirements.txt`, then use this process command:

```text
python dashboard_app.py worker
```

Configure the host supervisor to restart the process after failures. Monitor the SQS dead-letter queue and CloudWatch logs. The Vercel function can accept and track jobs while the worker is offline, but those jobs will remain queued until the worker returns.

## Behavior and Limits

- Folder uploads preserve relative subfolder paths. Multi-PDF uploads are treated as root-level files.
- Rename scans only root-level PDFs and writes copies named like `I 10009.pdf`. Duplicate detected IDs in one rename job are skipped rather than overwritten. Manual ID entry is also queued.
- Merge processes each PDF-containing subfolder independently and sorts filenames naturally. Corrupted PDFs or OCR failures become per-file/per-folder failures; processing continues where possible.
- Originals are always preserved. S3 lifecycle cleanup is the only automatic file deletion mechanism; DynamoDB TTL cleanup is asynchronous.
- Pause, resume, and stop persist in DynamoDB and take effect between PDF or subfolder processing units. An in-progress OCR or merge operation cannot be interrupted mid-file.
- The app currently uses a shared access token, not per-user accounts or tenant-level authorization. Keep the deployment private or place it behind an organization identity proxy if multiple users or sensitive records are involved.
- Large uploads bypass Vercel's request-body limit through S3, but processing capacity is limited by the external worker's disk, RAM, OCR model availability, SQS visibility timeout, and configured size limits.
- RapidOCR's model/runtime packaging must fit the selected Vercel Python bundle. If deployment build reports the function bundle is too large, move the OCR package out of the web function dependency set and keep it installed only in the worker environment.

## Validation

The Flask import and local dashboard response can be checked without AWS:

```powershell
python -m py_compile app.py dashboard_app.py
python -c "from app import app; from dashboard_app import app as existing; assert app is existing; print(app.name)"
```

Cloud upload, queue, OCR, processing, and signed-download verification require the configured AWS resources and a running worker.
