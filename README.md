# moneyball-ml
MLB statcast data vs conventional baseball metrics in win prediction

- [ ] Use pybaseball to fetch all 2024 stat cast AND traditional stats
    - https://github.com/jldbc/pybaseball and the refernece for the API (apparently)
    - https://baseballsavant.mlb.com/csv-docs

- [ ] Find the team wins, ideally time-anchored so the predictions can be linked to these 
- [ ] Explore the data available - Jupyter 
- [ ] Look at distributions as they pertain to the target 
- [ ] Decide whether time series is relevant and how we’d tackle it
- [ ] Filter anomalies, etc and subset as needed, reject goofy stats I can’t figure out how to make relevant or incorporate
- [ ] Borrow my pipeline from Kaggle comp and use that to check baseline performance
- [ ] Take a similar structured data prediction task and use that NN architecture 
- [ ] Figure out how to extend pipeline to support deep learning HPO w/ PyTorch 
- [ ] Document data and data prep pipeline
- [ ] Document metric selection and rationale 
- [ ] Document evaluation strategy 
- [ ] Document model optimization 
- [ ] Describe conventional and deep learning models
- [ ] Document thoughts on model performance 
