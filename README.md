# "Ondemand" Whisper with Fly.io Machines

Work in progress!

This server allows one to download a Youtube video and run Whisper on the audio, and output a transcript.

- Whisper flavor: https://github.com/Vaibhavs10/insanely-fast-whisper
- yt-dlp for downloading videos https://github.com/yt-dlp/yt-dlp
- FastAPI [Background Tasks](https://fastapi.tiangolo.com/tutorial/background-tasks/) + SQLite for queue

```bash
fly deploy -a $APP_NAME
```

TODO:

- [ ] Proper queue system (ie concurrent jobs)
- [ ] Scale to zero
