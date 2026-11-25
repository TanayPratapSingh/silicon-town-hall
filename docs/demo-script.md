# Demo script: 10 minute presentation

The deck is in [slides/](slides/) (11 slides, speaker notes on each). Times: goal and how it
works 3 minutes, live demo 4, key insights 2, questions 1.

| Slides | Time | What |
| --- | --- | --- |
| 1 to 5: cover, the problem, how a team uses it, the outreach brief, how it works | 3:00 | Goal and how it works |
| 6: live demo | 4:00 | The script below |
| 7 and 8: where simulated residents get people wrong, what it is reliable for | 2:00 | Key insights |
| 9 to 11: ethics, what comes next, questions | 1:00 | Questions |

## Before the demo

Open the app on the laptop (`streamlit run app/streamlit_app.py`), log in, leave Demo mode on, and keep a recorded
backup of this demo open in another window. Everything but the map tiles works offline.

## The four minutes

1. **0:00 to 0:45, Community.** I-81 Syracuse, 8 residents, Sample residents. Point at the tiles, then the Map tab:
   red neighborhoods have fewer simulated residents than the Census implies. Hover Northside: 0 simulated, 2.1
   expected. Say: this list rests on Census counts, not on model text.
2. **0:45 to 1:30, Run meetings.** Meet by: One citywide meeting. Try a change: "Construction near homes limited to
   7 am to 7 pm on weekdays." Run meetings. While it runs: demo mode is template text, so it costs nothing and every
   line is labeled simulated; with the real model the Cost tile shows the price first. Then Open the report.
3. **1:30 to 2:30, Report.** Pick "I-81 pilot: 20 residents, one citywide meeting". Who to reach: the shortfall chart
   and the map. Click Download outreach brief and open it: who to reach, concerns by neighborhood, sample comments
   with page citations, labeled simulated. Scroll to the end: "Check against the real I-81 comments" shows no
   simulated resident raised air quality, which 165 real comments raised.
4. **2:30 to 3:40, Saved scenarios.** Compare the I-81 control and the I-81 what if, both real runs on the same 20
   residents: 9 support in the control, 20 of 20 with the change, and every what if comment raises the change's own
   topics. Open Meeting replay on the what if and read one turn.
5. **3:40 to 4:00, close.** The tool is good at showing who is missing, and it shows how strongly simulated residents
   react to a change, which is itself a warning.

If the network fails, keep going: only the map tiles need it. If the app fails, play the recorded backup.
