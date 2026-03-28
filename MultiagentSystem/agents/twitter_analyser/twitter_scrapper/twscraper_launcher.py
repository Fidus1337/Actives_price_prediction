"""
Selenium-based Twitter scraper.

Opens real Chrome browser, logs into Twitter, navigates to user profiles,
and extracts tweets from the page DOM.

First run requires login (credentials from dev.env).
Subsequent runs reuse saved cookies from twitter_cookies.json.

Usage:
    python -m MultiagentSystem.agents.twitter_analyser.twitter_scrapper.twscraper_launcher
    python -m MultiagentSystem.agents.twitter_analyser.twitter_scrapper.twscraper_launcher --login
"""

import json
import os
import random
import re
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
import undetected_chromedriver as uc
from selenium.common.exceptions import StaleElementReferenceException, WebDriverException
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
load_dotenv(_PROJECT_ROOT / "dev.env")

COOKIES_PATH = Path(__file__).parent / "twitter_cookies.json"
LOG_TAG = "[twitter_selenium]"

_ARTICLE_SELECTOR = 'article[data-testid="tweet"]'
_TWEET_TEXT_SELECTOR = 'div[data-testid="tweetText"]'
_SOCIAL_CONTEXT_SELECTOR = 'span[data-testid="socialContext"]'
_NO_NEW_THRESHOLD = 5
_SCROLL_WAIT_BASE = 2.0
_SCROLL_WAIT_MAX = 8.0
_PAGE_LOAD_RETRIES = 3


def _init_driver(headless: bool = True) -> uc.Chrome:
    """Start undetected Chrome with a dedicated profile for scraping."""
    scraper_profile = Path(__file__).parent / "chrome_profile"
    scraper_profile.mkdir(exist_ok=True)

    options = uc.ChromeOptions()
    if headless:
        options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--lang=en-US")

    driver = uc.Chrome(options=options, user_data_dir=str(scraper_profile), version_main=146)
    return driver


def _save_cookies(driver: uc.Chrome) -> None:
    """Save browser cookies to JSON file."""
    cookies = driver.get_cookies()
    COOKIES_PATH.write_text(json.dumps(cookies, indent=2), encoding="utf-8")
    print(f"{LOG_TAG} Cookies saved ({len(cookies)} items)")


def _load_cookies(driver: uc.Chrome) -> bool:
    """Load cookies from file into browser. Returns True if loaded."""
    if not COOKIES_PATH.exists():
        return False

    cookies = json.loads(COOKIES_PATH.read_text(encoding="utf-8"))
    if not cookies:
        return False

    driver.get("https://x.com")
    time.sleep(5)

    for cookie in cookies:
        cookie.pop("sameSite", None)
        cookie.pop("expiry", None)
        try:
            driver.add_cookie(cookie)
        except Exception:
            pass

    print(f"{LOG_TAG} Cookies loaded ({len(cookies)} items)")
    return True


def _is_logged_in(driver: uc.Chrome) -> bool:
    """Check if we're logged into Twitter."""
    driver.get("https://x.com/home")
    time.sleep(6)
    url = driver.current_url
    logged_in = "login" not in url and "i/flow" not in url
    print(f"{LOG_TAG} Login check: url={url}, logged_in={logged_in}")
    return logged_in


def _login(driver: uc.Chrome) -> bool:
    """Login to Twitter via UI. Returns True on success."""
    email = os.getenv("TWITTER_EMAIL", "")
    username = os.getenv("TWITTER_USERNAME", "")
    password = os.getenv("TWITTER_PASSWORD", "")

    if not email or not password:
        print(f"{LOG_TAG} ERROR: Set TWITTER_EMAIL and TWITTER_PASSWORD in dev.env")
        return False

    print(f"{LOG_TAG} Logging in with {email}...")
    driver.get("https://x.com/i/flow/login")
    time.sleep(5)

    try:
        wait = WebDriverWait(driver, 20)

        username_input = wait.until(
            EC.presence_of_element_located((By.CSS_SELECTOR, 'input[autocomplete="username"]'))
        )
        username_input.clear()
        username_input.send_keys(email)
        username_input.send_keys(Keys.RETURN)
        time.sleep(3)

        unusual_activity = driver.find_elements(By.CSS_SELECTOR, 'input[data-testid="ocfEnterTextTextInput"]')
        if unusual_activity:
            print(f"{LOG_TAG} Verification required, trying @{username}...")
            unusual_activity[0].send_keys(username)
            unusual_activity[0].send_keys(Keys.RETURN)
            time.sleep(3)

            unusual_activity_retry = driver.find_elements(By.CSS_SELECTOR, 'input[data-testid="ocfEnterTextTextInput"]')
            if unusual_activity_retry:
                print(f"{LOG_TAG} Username rejected, trying email...")
                unusual_activity_retry[0].clear()
                unusual_activity_retry[0].send_keys(email)
                unusual_activity_retry[0].send_keys(Keys.RETURN)
                time.sleep(3)

        password_input = wait.until(
            EC.presence_of_element_located((By.CSS_SELECTOR, 'input[name="password"]'))
        )
        password_input.clear()
        password_input.send_keys(password)
        password_input.send_keys(Keys.RETURN)
        time.sleep(5)

        url = driver.current_url
        print(f"{LOG_TAG} Post-login URL: {url}")
        if "home" in url:
            print(f"{LOG_TAG} Login successful")
            _save_cookies(driver)
            return True
        else:
            print(f"{LOG_TAG} Login may have failed. URL: {url}")
            _save_cookies(driver)
            return "login" not in url and "i/flow" not in url

    except Exception as e:
        print(f"{LOG_TAG} Login error: {e}")
        return False


def _ensure_logged_in(driver: uc.Chrome) -> bool:
    """Load cookies or login. Returns True if logged in."""
    if _load_cookies(driver) and _is_logged_in(driver):
        print(f"{LOG_TAG} Session restored from cookies")
        return True

    print(f"{LOG_TAG} Cookies expired or missing, logging in...")
    return _login(driver)


def _parse_count(text: str) -> int:
    """Parse engagement count like '1.2K', '3.5M', '456'."""
    if not text:
        return 0
    text = text.strip().replace(",", "")
    match = re.match(r"([\d.]+)\s*([KMB]?)", text, re.IGNORECASE)
    if not match:
        return 0
    num = float(match.group(1))
    suffix = match.group(2).upper()
    multipliers = {"K": 1_000, "M": 1_000_000, "B": 1_000_000_000}
    return int(num * multipliers.get(suffix, 1))


def _is_driver_connection_error(exc: Exception) -> bool:
    """Return True when Selenium transport/session is broken."""
    msg = str(exc).lower()
    markers = (
        "winerror 10054",
        "connection reset",
        "forcibly closed by the remote host",
        "invalid session id",
        "disconnected",
        "unable to receive message from renderer",
        "failed to decode response from marionette",
    )
    return isinstance(exc, ConnectionResetError) or any(marker in msg for marker in markers)


def _get_page_height(driver: uc.Chrome) -> int:
    return driver.execute_script("return document.body.scrollHeight")


def _get_article_count(driver: uc.Chrome) -> int:
    return len(driver.find_elements(By.CSS_SELECTOR, _ARTICLE_SELECTOR))


def _wait_for_new_articles(
    driver: uc.Chrome, prev_count: int, timeout: float = 10.0
) -> bool:
    """Wait until DOM has more articles than prev_count, or timeout.

    Returns True if new articles appeared.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        if _get_article_count(driver) > prev_count:
            return True
        time.sleep(0.5)
    return False


def _expand_show_more(driver: uc.Chrome, article) -> None:
    """Click 'Show more' inside a tweet if the text is truncated."""
    try:
        show_buttons = article.find_elements(
            By.CSS_SELECTOR,
            'div[data-testid="tweet-text-show-more-link"], '
            'button[data-testid="tweet-text-show-more-link"]',
        )
        if not show_buttons:
            show_buttons = article.find_elements(By.XPATH, ".//span[text()='Show more']/..")
        for btn in show_buttons:
            try:
                driver.execute_script("arguments[0].click();", btn)
                time.sleep(0.5)
            except Exception:
                pass
    except Exception:
        pass


def _extract_tweet_text(article) -> str:
    """Extract full tweet text, handling cashtags, hashtags, emojis, and links.

    Falls back from child-element concatenation to plain .text.
    """
    text_el = article.find_elements(By.CSS_SELECTOR, _TWEET_TEXT_SELECTOR)
    if not text_el:
        return ""

    container = text_el[0]

    try:
        children = container.find_elements(By.XPATH, ".//span | .//a | .//img")
        if children:
            parts: list[str] = []
            for child in children:
                tag = child.tag_name
                if tag == "img":
                    alt = child.get_attribute("alt") or ""
                    if alt:
                        parts.append(alt)
                    continue
                child_text = child.text
                if child_text:
                    parts.append(child_text)
            assembled = " ".join(parts)
            assembled = re.sub(r"\s{2,}", " ", assembled).strip()
            if assembled:
                return assembled
    except Exception:
        pass

    try:
        return container.text.strip()
    except Exception:
        return ""


def _is_retweet(article, username: str) -> tuple[bool, str]:
    """Detect retweet using social context banner (more reliable than author link).

    Returns (is_retweet, original_author_username).
    """
    try:
        social_ctx = article.find_elements(By.CSS_SELECTOR, _SOCIAL_CONTEXT_SELECTOR)
        if social_ctx:
            ctx_text = social_ctx[0].text.lower()
            if "reposted" in ctx_text or "retweeted" in ctx_text:
                user_links = article.find_elements(
                    By.CSS_SELECTOR, 'div[data-testid="User-Name"] a[role="link"]'
                )
                for link in user_links:
                    href = link.get_attribute("href") or ""
                    author = href.rstrip("/").split("/")[-1]
                    if author.lower() != username.lower():
                        return True, author
                return True, username
    except Exception:
        pass

    return False, username


def _extract_tweets_from_page(driver: uc.Chrome, username: str) -> list[dict]:
    """Extract tweet data from currently loaded page.

    Handles: cashtags, emojis, Show more, retweet detection via social context,
    quoted tweets (not flagged as retweets), multiple time elements.
    """
    tweets = []
    seen_ids: set[str] = set()
    stale_skipped = 0

    try:
        articles = driver.find_elements(By.CSS_SELECTOR, _ARTICLE_SELECTOR)
    except Exception as e:
        if _is_driver_connection_error(e):
            raise RuntimeError("DRIVER_CONNECTION_LOST") from e
        print(f"{LOG_TAG}   Error getting articles: {type(e).__name__}: {str(e).splitlines()[0]}")
        return []

    for article in articles:
        try:
            time_elements = article.find_elements(By.CSS_SELECTOR, "time")
            if not time_elements:
                continue

            link_el = time_elements[0].find_element(By.XPATH, "./..")
            tweet_url = link_el.get_attribute("href") or ""

            tweet_id_match = re.search(r"/status/(\d+)", tweet_url)
            if not tweet_id_match:
                continue
            tweet_id = tweet_id_match.group(1)

            if tweet_id in seen_ids:
                continue
            seen_ids.add(tweet_id)

            datetime_str = time_elements[0].get_attribute("datetime") or ""

            _expand_show_more(driver, article)

            text = _extract_tweet_text(article)

            is_rt, author_username = _is_retweet(article, username)

            display_name = ""
            if not is_rt:
                author_username = username
            user_name_el = article.find_elements(
                By.CSS_SELECTOR, 'div[data-testid="User-Name"] span'
            )
            if user_name_el:
                display_name = user_name_el[0].text

            reply_el = article.find_elements(By.CSS_SELECTOR, 'button[data-testid="reply"] span')
            retweet_el = article.find_elements(By.CSS_SELECTOR, 'button[data-testid="retweet"] span')
            like_el = article.find_elements(By.CSS_SELECTOR, 'button[data-testid="like"] span')
            views_el = article.find_elements(By.CSS_SELECTOR, 'a[href$="/analytics"] span')

            date_str = ""
            created_at = ""
            if datetime_str:
                try:
                    dt = datetime.fromisoformat(datetime_str.replace("Z", "+00:00"))
                    date_str = dt.strftime("%Y-%m-%d")
                    created_at = dt.isoformat()
                except ValueError:
                    pass

            tweet_dict = {
                "tweet_id": tweet_id,
                "author_username": author_username,
                "author_display_name": display_name,
                "text": text,
                "created_at": created_at,
                "date": date_str,
                "likes": _parse_count(like_el[0].text if like_el else "0"),
                "retweets": _parse_count(retweet_el[0].text if retweet_el else "0"),
                "replies": _parse_count(reply_el[0].text if reply_el else "0"),
                "views": _parse_count(views_el[0].text if views_el else "0"),
                "is_retweet": is_rt,
                "is_reply": False,
                "lang": "",
                "url": tweet_url,
            }
            tweets.append(tweet_dict)

        except StaleElementReferenceException:
            stale_skipped += 1
            continue
        except Exception as e:
            if _is_driver_connection_error(e):
                raise RuntimeError("DRIVER_CONNECTION_LOST") from e
            err_line = str(e).splitlines()[0] if str(e) else ""
            print(f"{LOG_TAG}   Error parsing article: {type(e).__name__}: {err_line}")
            continue

    if stale_skipped:
        print(f"{LOG_TAG}   Skipped stale articles: {stale_skipped}")

    return tweets


def create_driver(headless: bool = True) -> uc.Chrome:
    """Create and authenticate a Chrome driver for reuse across multiple accounts.

    Returns an authenticated driver. Caller is responsible for driver.quit().
    """
    driver = _init_driver(headless=headless)
    if not _ensure_logged_in(driver):
        driver.quit()
        raise RuntimeError("Could not log in to Twitter")
    return driver


def _load_profile_with_retry(
    driver: uc.Chrome, username: str, retries: int = _PAGE_LOAD_RETRIES
) -> bool:
    """Navigate to user profile and wait for tweets to appear.

    Retries on failure (page might load without tweets on first attempt
    due to rate limiting or network issues).
    """
    for attempt in range(1, retries + 1):
        driver.get(f"https://x.com/{username}")
        try:
            WebDriverWait(driver, 20).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, _ARTICLE_SELECTOR))
            )
            time.sleep(random.uniform(1.5, 3.0))
            article_count = _get_article_count(driver)
            print(f"{LOG_TAG} Page loaded for @{username}: {article_count} articles (attempt {attempt})")
            return True
        except Exception:
            print(f"{LOG_TAG} WARNING: No tweets on attempt {attempt}/{retries} for @{username}")
            if attempt < retries:
                time.sleep(random.uniform(3, 6))

    return False


def _smart_scroll(driver: uc.Chrome) -> None:
    """Scroll down by ~1 viewport height instead of jumping to absolute bottom.

    This better mimics human scrolling and more reliably triggers
    Twitter's infinite-scroll content loading.
    """
    driver.execute_script(
        "window.scrollBy(0, Math.max(window.innerHeight * 0.85, 600));"
    )


def _detect_end_of_timeline(driver: uc.Chrome) -> bool:
    """Check if Twitter shows 'end of timeline' indicators."""
    try:
        page_text = driver.execute_script(
            "return document.body.innerText.substring(document.body.innerText.length - 500);"
        )
        end_markers = [
            "these highlights don't exist yet",
            "nothing to see here",
            "doesn't have any",
        ]
        lower_text = page_text.lower()
        return any(marker in lower_text for marker in end_markers)
    except Exception:
        return False


def fetch_tweets_sync(
    username: str,
    max_tweets: int | None = 20,
    since_date: str | None = None,
    until_date: str | None = None,
    driver: uc.Chrome | None = None,
    max_scrolls: int = 100,
) -> list[dict]:
    """Fetch recent tweets from a Twitter user profile.

    Args:
        username: Twitter handle without @
        max_tweets: Max tweets to collect. None = unlimited (bounded by max_scrolls).
        since_date: "YYYY-MM-DD" — inclusive lower bound (tweets ON this date are kept)
        until_date: "YYYY-MM-DD" — inclusive upper bound (tweets ON this date are kept)
        driver: Reusable Chrome driver (created via create_driver()).
                If None, creates and closes a temporary one.
        max_scrolls: Safety limit on page scrolls (default 100).

    Returns:
        List of tweet dicts ready for archiving.
        Date filtering: [since_date, until_date] — both bounds inclusive.
    """
    own_driver = driver is None
    if own_driver:
        driver = _init_driver(headless=True)
    assert driver is not None

    try:
        if own_driver and not _ensure_logged_in(driver):
            print(f"{LOG_TAG} ERROR: Could not log in to Twitter")
            return []

        print(f"{LOG_TAG} Navigating to @{username} (since={since_date}, until={until_date})...")

        if not _load_profile_with_retry(driver, username):
            print(f"{LOG_TAG} ERROR: Could not load profile for @{username}")
            return []

        all_tweets: list[dict] = []
        seen_ids: set[str] = set()
        no_new_count = 0
        consecutive_old_only = 0

        for scroll_i in range(max_scrolls):
            prev_article_count = _get_article_count(driver)
            prev_height = _get_page_height(driver)

            page_tweets = _extract_tweets_from_page(driver, username)

            new_count = 0
            old_count = 0
            stop_scrolling = False

            for t in page_tweets:
                if t["tweet_id"] in seen_ids:
                    continue
                seen_ids.add(t["tweet_id"])

                if since_date and t["date"] and t["date"] < since_date:
                    old_count += 1
                    continue

                if until_date and t["date"] and t["date"] > until_date:
                    continue

                all_tweets.append(t)
                new_count += 1

                if max_tweets is not None and len(all_tweets) >= max_tweets:
                    stop_scrolling = True
                    break

            if scroll_i % 5 == 0 and scroll_i > 0:
                print(
                    f"{LOG_TAG}   scroll {scroll_i}: "
                    f"{len(all_tweets)} collected, {new_count} new this pass, "
                    f"{old_count} old, {len(seen_ids)} seen total"
                )

            if old_count > 0 and new_count == 0:
                consecutive_old_only += 1
                if consecutive_old_only >= 3:
                    print(f"{LOG_TAG}   Stopping: 3 consecutive scrolls with only old tweets")
                    break
            else:
                consecutive_old_only = 0

            if stop_scrolling:
                break

            if _detect_end_of_timeline(driver):
                print(f"{LOG_TAG}   Reached end of timeline")
                break

            if new_count == 0 and old_count == 0:
                no_new_count += 1
                if no_new_count >= _NO_NEW_THRESHOLD:
                    print(f"{LOG_TAG}   Stopping: {_NO_NEW_THRESHOLD} scrolls with no new tweets")
                    break
            else:
                no_new_count = 0

            _smart_scroll(driver)

            new_articles_appeared = _wait_for_new_articles(
                driver, prev_article_count, timeout=8.0
            )

            if not new_articles_appeared:
                new_height = _get_page_height(driver)
                if new_height == prev_height:
                    extra_wait = random.uniform(2, 4)
                    time.sleep(extra_wait)
                    driver.execute_script(
                        "window.scrollTo(0, document.body.scrollHeight);"
                    )
                    time.sleep(random.uniform(2, 4))
                else:
                    time.sleep(random.uniform(1, 2))
            else:
                time.sleep(random.uniform(0.5, 1.5))

        print(
            f"{LOG_TAG} Done: {len(all_tweets)} tweets from @{username} "
            f"({len(seen_ids)} seen, {scroll_i + 1} scrolls)"
        )
        return all_tweets

    finally:
        if own_driver:
            driver.quit()


def manual_login() -> None:
    """Open Chrome for manual login. User logs in, script saves cookies."""
    driver = _init_driver(headless=False)
    try:
        driver.get("https://x.com/i/flow/login")
        print(f"\n{LOG_TAG} Chrome открыт. Залогинься вручную в Twitter.")
        print(f"{LOG_TAG} После логина (когда увидишь ленту) нажми Enter здесь...")
        input()

        if "home" in driver.current_url or "login" not in driver.current_url:
            _save_cookies(driver)
            print(f"{LOG_TAG} Cookies сохранены! Теперь скрейпер будет работать без логина.")
        else:
            print(f"{LOG_TAG} Похоже логин не прошёл. URL: {driver.current_url}")
    finally:
        driver.quit()


if __name__ == "__main__":
    if "--login" in sys.argv:
        manual_login()
