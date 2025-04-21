# moneyball-ml

Major League Baseball (MLB) has made a shift toward data-driven decision-making in the past 20 years. Initially driven by more traditional statistics that could be compiled by a human scorekeeper, the trend in recent years has been to embrace machine-captured StatCast data to evaluate players. However, many of the statistics cited by baseball enthusiasts and commentators remain those which can be derived from the traditional statistics, and not their richer StatCast counterparts. For example weighted overall batting average aka wOBA, a common metric for hitting performance, is an amalgamation of at-bats, singles, doubles, walks, etc as indicated by the below definition:

wOBA flash 
      Graphic source: FanGraphs, wOBA definition, see https://library.fangraphs.com/offense/woba/

Furthermore, StatCast data is typically used to assess a player's potential, independent or in conjunction with conventional metrics. As extensive information on historical MLB seasons and its players have been compiled, it is possible to analyze StatCast data for its predictive value in the context of other important outcomes such as win rate and run differential. Notably its predictive value here can be compared against the now conventional statistics that rely on the non-Statcast-sourced data. I propose to analyze and model the relationship to predict game outcomes with inputs that are: 
- Strictly sourced from conventional, non-Statcast reporting methods
- Strictly sourced from Statcast
- A combination of A and B above

In so doing, I hope to shed light on the relative value of the newer, more expensive, and often-touted metric source. The data that underpins the proposed analysis is available on the MLB website, a partial list of relevant datasets is provided here: 
- Pitch tempo: https://baseballsavant.mlb.com/leaderboard/pitch-tempo
- Pitch movement: https://baseballsavant.mlb.com/leaderboard/pitch-movement
- Bat tracking : https://baseballsavant.mlb.com/leaderboard/bat-tracking
- Batted ball profile: https://baseballsavant.mlb.com/leaderboard/batted-ball

While the structured nature of these datasets supports classical machine learning (ML) methods, the volume and complexity also provide an interesting testbed for neural-network-based ML algorithms.

# Repository 

The moneyball-ml application includes a command-line interface to kick off the data and modeling pipeline. 
- [mb.py](mb.py): CLI, model pipeline, validation, visualization
- [nn.py](nn.py): Neural network implementation, storage
- [data.py](data.py): Data ingest, cleaning, feature implementation, normalization, caching
- [requirements.txt](requirements.txt): Python package dependencies

# Usage 

A virtual environment is recommended. Python 3.13 and Skorch are not yet compatible. Tested with Python 3.12. 

The app will print usage, to run the data and modeling pipeline for a full grid search: 

```
usage: mb.py [-h] [-s | --search | --no-search | -e | --evaluate | --no-evaluate] [-k SPLITS] [-t THRESHOLD] [--conventional | --no-conventional] [--statcast | --no-statcast] [--visualize | --no-visualize]

options:
  -h, --help            show this help message and exit
  -s, --search, --no-search
  -e, --evaluate, --no-evaluate
  -k SPLITS, --splits SPLITS
  -t THRESHOLD, --threshold THRESHOLD
  --conventional, --no-conventional
  --statcast, --no-statcast
  --visualize, --no-visualize
``` 
