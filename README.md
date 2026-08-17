# CohesionNN

Tim again. This was my first attempt, before the [shot-quality model](https://github.com/elvira-T52/tbl-shot-quality-RNN). It's roughly the same idea (an RNN reading a sequence of hockey plays), a much harder question however, and honestly the project that taught me why "predict the next event in a hockey game" is a lot harder than it seems.

## What I was trying to do

CohesionNN watched the plays building up around six Tampa Bay players' lines and tried to predict two things at once for whatever play came next: **who** on the line would make it (a classification across roughly 65 players) and **what kind of play** it would be (goal, shot, hit, faceoff, takeaway...). The idea was to get at "line cohesion" — whether certain players make their linemates more productive — by seeing how well a model could anticipate what a line does together.

## What failed

The core problem: "what happens next in a live hockey sequence" is genuinely high-entropy. Unlike a single shot's outcome (a well-scoped yes/no question), the very next event after any given play could reasonably be any of 8 different things, done by any of several players on the ice — and a lot of that really is close to random. Concretely:

- **It struggled to converge.** Validation loss hit its best point somewhere around epoch 3-7 out of 20 in every run I tried, then climbed steadily afterward no matter how much data I threw at it. Even then though, the validation loss was never good enough for a model to predict comfortably above 22% playtype accuracy.
- **More data didn't raise the ceiling much.** I went from one player (Brandon Hagel) (~8,300 training sequences) to six players across four seasons (~45,800 sequences), a 5.5x increase, and best-case accuracy barely moved: about 26% for play type and 22-23% for who acts, against a baseline of ~22% from just always guessing the single most common play type. Adding more data didn't improve the performance, as the problem lies in what was learned: hockey plays are highly chaotic and difficult to predict.
- **Class weighting traded one problem for another.** Goals and rare event types are heavily outnumbered by faceoffs and shots-on-goal. Weighting the loss to compensate helped balance the model's attention across classes, but tanked raw accuracy, since it could no longer coast on the easy majority classes.

## Performance

Best run (six players, four seasons, no class weighting):

| Metric | Peak value |
|---|---|
| Validation loss | 4.42 (epoch 4 — climbed to 4.6+ by epoch 12) |
| Play-type accuracy | 26.2% (epoch 3) vs. ~22% baseline |
| Who-acts accuracy | 22-23% (early epochs) |


## What data would have actually helped: passing

The NHL's public play-by-play doesn't log passes at all — only discrete events like shots, hits, faceoffs, giveaways, and takeaways. That means the model never sees most of what actually happens on the ice; a string of passes moving the puck into a dangerous area is completely invisible to it, and all it ever gets is the plays on either side of that gap. Given the question I was actually asking: what happens next and who does it; pass-by-pass data (who passed to whom, from where, completed or intercepted) is probably the single biggest missing piece. It's exactly the connective tissue a sequence model needs to actually learn "the play developing," instead of a series of disconnected snapshots with everything in between removed.
