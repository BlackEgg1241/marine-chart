Rescore all GPS catch dates (from Export.csv via validate_scoring.py) and today + 6 forecast dates.

Run: `python rescore_quick.py` from the project root.

This rescores:
1. The ~46 catch dates that have GPS coordinates in Export.csv (the validation set)
2. Today + 6 days of forecast/prediction dates

It does NOT rescore all 96+ observation dirs or old backtest dirs — only the dates that matter for validation and the current forecast.

After rescoring, run `python validate_scoring.py` to check validation metrics and report the results.
