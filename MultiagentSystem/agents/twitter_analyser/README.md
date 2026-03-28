# Twitter Analyser Agent

## Purpose
Collects tweets from configured accounts, classifies BTC signal, and stores only directional items (`BULL` or `BEAR`) in SQLite archive.

## Main files
- `twitter_scrapper/twitter_collector.py` - collection entrypoint and run modes
- `twitter_classifier.py` - LLM tweet signal classifier
- `twitter_scrapper/twscraper_launcher.py` - Selenium/X scraping logic
- `twitter_collector_settings.json` - accounts and collection window settings

## How to run
- Fetch new tweets and classify:
  - `python -m MultiagentSystem.agents.twitter_analyser.twitter_scrapper.twitter_collector --fetch-new`
- Reclassify all tweets currently in DB:
  - `python -m MultiagentSystem.agents.twitter_analyser.twitter_scrapper.twitter_collector --reclassify-all`
- Reclassify all and refetch for configured date range:
  - `python -m MultiagentSystem.agents.twitter_analyser.twitter_scrapper.twitter_collector --reclassify-and-refetch`

## Inputs and outputs
- Input source: X/Twitter profile pages (Selenium)
- Archive DB: `twitter_archive.db`
- Session cookies: `twitter_scrapper/twitter_cookies.json`

