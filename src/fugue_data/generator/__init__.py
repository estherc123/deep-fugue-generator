"""Plan-conditioned, time-aware fugue generator (generator track, v1).

Each token is one voice's decision at one grid step: NOTE_ON_<midi>, HOLD or REST.
Musical time and the form plan are not tokens but per-position feature
embeddings added to the token embedding, so the context window is spent on music
and every position knows where it is in the bar, in the piece and in the plan.
"""
