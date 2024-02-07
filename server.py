import yt_dlp
from pathlib import Path
import json
import sqlite3
import sqlite3
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import jinja2
from os import environ
import torch
from transformers import pipeline
from transformers.utils import (
    is_flash_attn_2_available,
)  # is false on fly.io GPU machine


MP3_DIRECTORY = environ["MP3_DIRECTORY"]
DB_PATH = environ["DB_PATH"]

db = sqlite3.connect(DB_PATH, check_same_thread=False)
db.row_factory = sqlite3.Row

with open("schema.sql") as f:
    db.executescript(f.read())

pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-large-v3",  # select checkpoint from https://huggingface.co/openai/whisper-large-v3#model-details
    torch_dtype=torch.float16,
    device="cuda:0",  # or mps for Mac devices
    model_kwargs={"attn_implementation": "flash_attention_2"}
    if is_flash_attn_2_available()
    else {"attn_implementation": "sdpa"},
)


class Queries:
    INSERT_JOB = """
      INSERT INTO jobs(id, status, job_type, url)
      VALUES (?, 'queue-yt', 'youtube', ?)
    """

    ENQUEUE_YT_DOWNLOADS = """
      INSERT INTO queue_yt_downloads(job_id, url, outtmpl)
      VALUES (?, ?, ?)
    """

    DEQUEUE_YT_DOWNLOADS = """
      SELECT
        job_id,
        url,
        outtmpl
      FROM queue_yt_downloads
      ORDER BY queued_at
      LIMIT 1;
    """

    DEQUEUE_WHISPER = """
      SELECT
        job_id,
        input_path
      FROM queue_whisper_transcripts
      ORDER BY queued_at
      LIMIT 1;
    """

    DELETE_YT_DOWNLOADS = "DELETE FROM queue_yt_downloads WHERE job_id = ?"

    ENQUEUE_RUNNING_YT_DOWNLOADS = """
      INSERT INTO queue_running_yt_downloads(job_id, url, outtmpl)
      VALUES (?, ?, ?)
    """
    DELETE_QUEUE_RUNNING_YT_DOWNLOADS = (
        "DELETE FROM queue_running_yt_downloads WHERE job_id = ?"
    )

    UPDATE_JOB_STATUS = "UPDATE jobs SET status = ? WHERE id = ?"
    UPDATE_JOB_FAILED = "UPDATE jobs SET status = 'failed', meta = ? WHERE id = ?"

    INSERT_QUEUE_WHISPER = """
      INSERT INTO queue_whisper_transcripts(job_id, input_path)
      VALUES (?, ?)
    """
    DELETE_QUEUE_WHISPER = "DELETE FROM queue_whisper_transcripts WHERE job_id = ?"

    INSERT_QUEUE_RUNNING_WHISPER = """
      INSERT INTO queue_running_whisper_transcripts(job_id, input_path)
      VALUES (?, ?)
    """
    DELETE_QUEUE_RUNNING_WHISPER = (
        "DELETE FROM queue_running_whisper_transcripts WHERE job_id = ?"
    )

    UPDATE_JOB_COMPLETE = """
      UPDATE jobs
      SET
        completed_at = CURRENT_TIMESTAMP,
        status = 'completed',
        transcript = ?
      WHERE id = ?
      """


class Queue:
    def __init__(self, db: sqlite3.Connection):
        self.db = db

    def init_job(self, job_id: str, url: str):
        outtmpl = str((Path(MP3_DIRECTORY) / f"{job_id}.%(ext)s").absolute())

        with self.db:
          self.db.execute(
              Queries.INSERT_JOB,
              [job_id, url],
          )
          self.db.execute(
              Queries.ENQUEUE_YT_DOWNLOADS,
              [job_id, url, outtmpl],
          )

    def mark_job_failed(self, job_id: str, message:str):
        with self.db:
          self.db.execute(
              Queries.UPDATE_JOB_FAILED,
              [message, job_id],
          )

    def dequeue_yt_download(self):
        with self.db:
          row = self.db.execute(Queries.DEQUEUE_YT_DOWNLOADS).fetchone()
          if row is None:
              return None
          job_id, url, outtmpl = row
          self.db.execute(Queries.DELETE_YT_DOWNLOADS, [job_id])
          self.db.execute(
              Queries.ENQUEUE_RUNNING_YT_DOWNLOADS, [job_id, url, outtmpl]
          )
          self.db.execute(Queries.UPDATE_JOB_STATUS, ["yt", job_id])
        return (job_id, url, outtmpl)

    def complete_yt_download(self, job_id, output_path):
        with self.db:
          self.db.execute(Queries.DELETE_QUEUE_RUNNING_YT_DOWNLOADS, [job_id])
          self.db.execute(Queries.INSERT_QUEUE_WHISPER, [job_id, output_path])
          self.db.execute(Queries.UPDATE_JOB_STATUS, ["queue-whisper", job_id])

    def dequeue_whisper(self):
        with self.db:
          row = self.db.execute(Queries.DEQUEUE_WHISPER).fetchone()
          if row is None:
              return None
          job_id, input_path = row
          self.db.execute(Queries.DELETE_QUEUE_WHISPER, [job_id])
          self.db.execute(Queries.INSERT_QUEUE_RUNNING_WHISPER, [job_id, input_path])
          self.db.execute(Queries.UPDATE_JOB_STATUS, ["whisper", job_id])
        return (job_id, input_path)

    def complete_whisper(self, job_id, transcript):
        with self.db:
          self.db.execute(Queries.DELETE_QUEUE_RUNNING_WHISPER, [job_id])
          self.db.execute(Queries.UPDATE_JOB_COMPLETE, [json.dumps(transcript), job_id])


def download_yt_mp3(url: str, outtmpl_target: str):
    ydl_opts = {
        "format": "m4a/bestaudio/best",
        # ℹ️ See help(yt_dlp.postprocessor) for a list of available Postprocessors and their arguments
        "postprocessors": [
            {  # Extract audio using ffmpeg
                "key": "FFmpegExtractAudio",
                "preferredcodec": "m4a",
                # "preferredcodec": "mp3",
            }
        ],
        "outtmpl": {"default": outtmpl_target},
        "quiet": True,
        "logtostderr": False,
        "verbose": False,
        "progress": False,
        'noprogress': True
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        # ydl.add_progress_hook(_hook)
        info = ydl.extract_info(url)
        m4a_filepath = info["requested_downloads"][0]["filepath"]

    return m4a_filepath

def _hook(status):
    print("\n")
    print(status.keys())
    print(status["status"])
    if status["status"] == "downloading":
        print("DOWNLOADING")
        print(status["filename"])
        print(status["elapsed"])
        print(status["eta"])
        print(status.get("downloaded_bytes"))
        print(status.get("total_bytes"))
    elif status["status"] == "finished":
        print("FINISHED")
        print(status["total_bytes"])
        print(status["filename"])
    else:
        print("ERROR")
        print(status)
    # print(status['downloaded_bytes'])
    # print(status['total_bytes'])
    # print(status['eta'])
    # print(status['elapsed'])
    print("\n")


queue = Queue(db)

app = FastAPI()


environment = jinja2.Environment()
template = environment.from_string(
    """
    <body>
     <div>
      <h2>Completed Jobs</h2>
      <table>
        <thead><tr></tr></thead>
        <tbody>

        </tbody>
      </table>


      <h2>Jobs</h2>
      <table>
        <thead>
          <tr>
            <th>ID</th>
            <th>submitted_at</th>
            <th>completed_at</th>
            <th>status</th>
            <th>url</th>
            <th>transcript</th>
          </tr>
        </thead>
        <tbody>
          {% for job in jobs %}
            <tr>
              <td>{{ job.id}}</td>
              <td>{{ job.submitted_at}}</td>
              <td>{{ job.completed_at}}</td>
              <td>{{ job.status}}</td>
              <td>{{ job.url}}</td>
              <td style="max-height: 100px;overflow: auto;display: block;">{{ job.transcript}}</td>
            </tr>
          {% else %}
            <tr><td colspan=10>No jobs!</td></tr>
          {% endfor %}
        </tbody>
      </table>
      <div>
        <h2>Debug</h2>
        <input id=url_input type="text"/>
        <button id=submit_button>Submit</button>
        <script>
          function submit() {
            let url;
            try{
              url = new URL(url_input.value);
            }catch {
              alert("Not a valid URL");
              return;
            }
            const id  = Date.now().toString();
            return fetch('/submit_job', {
              method: "POST",
              headers: {'Content-Type': 'application/json'},
              body: JSON.stringify({
                id,
                url: url.toString()
              })
            });
          }
          submit_button.addEventListener('click', () => submit().then(() => {
            submit_button.textContent = '👍';
          }))

        </script>
      </div>
     </div>
    </body>
    """
)


@app.get("/", response_class=HTMLResponse)
def read_root():
    jobs = queue.db.execute("select * from jobs").fetchall()
    completed_jobs = queue.db.execute("select * from jobs where status = 'completed'").fetchall()
    return template.render(
      jobs=jobs,
      completed_jobs=completed_jobs,
    )


def process_job():
    print("processing")
    while True:
        next = queue.dequeue_yt_download()
        if next is None:
            print("YT download queue empty")
            break
        job_id, url, outtmpl = next
        print("starting download for ", job_id, url)
        try:
            output_path = download_yt_mp3(url, outtmpl)
            queue.complete_yt_download(job_id, output_path)
        except Exception as e:
            queue.mark_job_failed(job_id, f"Exception downloading video: {e}")

        while True:
            next = queue.dequeue_whisper()
            if next is None:
                print("Whisper queue empty")
                break
            job_id, input_path = next
            print("Starting whisper for", job_id)
            try:
                transcript = pipe(
                    input_path,
                    chunk_length_s=30,
                    batch_size=24,
                    return_timestamps=True,
                )

                queue.complete_whisper(job_id, transcript)
                Path(input_path).unlink()
            except Exception as e:
                queue.mark_job_failed(job_id, f"Exception transcribing video: {e}")


class JobRequest(BaseModel):
    id: str
    url: str


@app.post("/submit_job")
def submit_job(job_request: JobRequest, background_tasks: BackgroundTasks):
    print(job_request)
    queue.init_job(job_request.id, job_request.url)
    background_tasks.add_task(process_job)

    return {"status": "queued", "job_id": job_request.id}


@app.get("/status/{job_id}")
def job_status(job_id: str):
    result = db.execute("select * from jobs where id = ?", [job_id]).fetchone()
    if result is None:
        raise HTTPException(status_code=404, detail="job not found")
    if result["status"] == "failure":
        return {"completed": True, "error": True, "message": result["meta"]}
    if result["status"] == "completed":
        return {"completed": True, "error": False, "transcript": json.loads(result["transcript"])}
    else:
        return {"completed": False, "error": False, "stage": result["status"]}
