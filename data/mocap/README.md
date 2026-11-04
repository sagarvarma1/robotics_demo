# CMU motion capture data

Files in `raw/` were downloaded unmodified from the CMU Graphics Lab Motion Capture Database
(plain HTTP, because HTTPS on that site has a certificate problem):

| file | URL | content |
|---|---|---|
| `raw/16.asf` | http://mocap.cs.cmu.edu/subjects/16/16.asf | skeleton of subject 16 |
| `raw/16_35.amc` | http://mocap.cs.cmu.edu/subjects/16/16_35.amc | trial 35, straight run/jog, 162 frames @ 120 fps |
| `raw/16_36.amc` | http://mocap.cs.cmu.edu/subjects/16/16_36.amc | trial 36, spare run, 189 frames @ 120 fps |

Rebuild the reference motions with `uv run python -m motion.build_cmu` (see `docs/notes_motion.md`).

The files are only parsed as text (`motion/cmu_amc.py`). Nothing in them is executed.

Acknowledgment (required by the database's terms of use):

The data used in this project was obtained from mocap.cs.cmu.edu. The database was created with funding from NSF EIA-0196217.

SHA-256 at download time (2026-11-04):

```
2323f876564610f84bfbec9b90b8ebffb57515673b7f4a45b0fb0849af465bdb  raw/16.asf
28e83b8ee8ff46e15cb7d41d5e77faaefe3e06cf165f05fb18dd348457bdf387  raw/16_35.amc
d0b511189db92f754c63a3cf0d93b71ebd67cea8b657f57d722c48b9896ba5a9  raw/16_36.amc
```
