/*

1) Submit job
  a) insert into jobs, status = 'queue-yt'
  b) insert into queue_yt_downloads
2) process queue_yt_downloads
  a) get first in queue_yt_downloads, delete it from queue_yt_downloads
  b) insert into queue_running_yt_downloads
  c) update jobs set status = 'yt'
3) on yt-dlp complete
  a) update jobs set status = 'whisper'
  b) delete from queue_running_yt_downloads
  c) insert into queue_whisper_transcripts
3) process queue_whisper_transcripts
  a) get first in queue_whisper_transcripts, delete from queue_whisper_transcripts
  b) insert into queue_running_whisper_transcripts
  c) update jobs set status = 'whisper'
4) on whisper complete
  a) update jobs set status = 'completed', transcript = :transcript
  b) delete from queue_running_whisper_transcripts
  c) on commit, delete input_mp3_path

*/

create table if not exists jobs(
  id text primary key,

  -- when the job was first submitted
  submitted_at datetime not null default CURRENT_TIMESTAMP,

  -- when status == 'completed', timestamp when fully complete
  completed_at datetime,

  download_started_at datetime,
  download_completed_at datetime,

  whisper_started_at datetime,
  whisper_completed_at datetime,

  video_title text,
  video_duration_seconds float,


  -- Only 'youtube' for now
  job_type text not null,

  -- 'queue-yt' | 'yt' | 'queue-whisper' | 'whisper' | 'completed' | 'failure'
  status text not null,

  -- when job_type == 'youtube', URL to youtube video
  url text not null,

  -- when status == 'completed', JSON transcript of the video
  transcript json,

  -- when status == 'failed', an error message
  meta text
);


create table if not exists queue_yt_downloads(
  job_id text references jobs(id),
  queued_at datetime,
  url text not null,
  outtmpl text not null,
  UNIQUE (job_id)
);
create table if not exists queue_running_yt_downloads(
  job_id text references jobs(id),
  started_at datetime default CURRENT_TIMESTAMP,
  url text not null,
  outtmpl text not null,
  UNIQUE (job_id)
);

create table if not exists queue_whisper_transcripts(
  job_id text references jobs(id),
  queued_at datetime default CURRENT_TIMESTAMP,
  input_path text not null,
  UNIQUE(job_id)
);
create table if not exists queue_running_whisper_transcripts(
  job_id text references jobs(id),
  submitted_at datetime default CURRENT_TIMESTAMP,
  input_path text not null,
  UNIQUE(job_id)
);
