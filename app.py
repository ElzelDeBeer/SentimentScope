"""
Sentiment Analysis & Data Insights Tool
---------------------------------------
Upload any CSV file that contains text (reviews, survey answers, patient
feedback, social media comments...) and the tool will:

  1. Score the sentiment of every row (Positive / Neutral / Negative) using VADER
  2. Show overall results, themes, group comparisons and trends over time
  3. Write plain-English insights automatically
  4. Let you download the scored data and the insights

Run it on your own computer:
    pip install -r requirements.txt
    streamlit run app.py

PRIVACY NOTE: Do not upload real personal or patient information that
identifies people. The built-in sample data is entirely fictional.
"""

import csv
import io
import re
import warnings
from collections import Counter
from html import escape

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from art import make_hero

# ----------------------------------------------------------------------
# SETTINGS
# ----------------------------------------------------------------------
POSITIVE_CUTOFF = 0.05   # standard VADER thresholds
NEGATIVE_CUTOFF = -0.05
MIN_MENTIONS = 3         # ignore themes/groups with fewer rows than this in the insights
ORDER = ["Positive", "Neutral", "Negative"]

# Colour palette: deep ink teal, with orange and mint as the data colours
POS, NEU, NEG = "#7FD8BE", "#8FA3AD", "#F2A31B"   # mint (positive), slate (neutral), orange (negative)
COLOURS = [POS, NEU, NEG]                          # Positive, Neutral, Negative
ACCENT = "#F6EFC9"                                 # cream
MUTED = "#9DB5B8"                                  # soft teal-grey for chart text
SENT_SCALE = alt.Scale(domain=ORDER, range=COLOURS)

# Themes are editable in the app. Each line is "Theme name: keyword, keyword, ..."
# Keywords match the START of a word, so "wait" also matches "waited" and "waiting".
DEFAULT_THEMES = """Waiting Time: wait, queue, delay, slow, quick, fast, on time, hour, minutes
Staff: nurse, doctor, staff, receptionist, pharmacist, dentist, team, midwi, technician
Facilities: clean, dirty, toilet, room, area, chair, seat, parking, building, crowded, noisy, bed
Communication: explain, inform, update, communicat, call, phone, email, sms, message, remind, told, listen
Service Quality: care, treat, service, quality, recommend, result, medication
Appointment Experience: appointment, book, reschedul, cancel, online, website"""

STOPWORDS = set(
    """the and was were that this with for have had has but not you your they them their there
    from are our out all very just can will would could about when what which who how why also
    than then into over after before while been being did does done get got its it's i'm i've
    one two too any some more most much many only own same such own off again further once
    here where because between both each few other these those through during above below
    up down in on at by of to is it as be or an if so we me my he she him her his us""".split()
)

SAMPLE_DATA = pd.DataFrame(
    {
        "date": [
            "2026-03-02", "2026-03-04", "2026-03-09", "2026-03-12", "2026-03-17", "2026-03-20",
            "2026-03-25", "2026-03-30", "2026-04-02", "2026-04-06", "2026-04-08", "2026-04-10",
        ],
        "department": [
            "Outpatients", "Emergency", "Outpatients", "Radiology", "Pharmacy", "Maternity",
            "Dental", "Outpatients", "Emergency", "Dental", "Pharmacy", "Maternity",
        ],
        "comment": [
            "I waited almost three hours before anyone called my name. The nurse was kind and explained everything clearly.",
            "Nobody told us why the doctor was delayed. The communication was terrible and we felt ignored.",
            "The toilets were dirty and the seating area was cramped and uncomfortable.",
            "I received my results quickly and the doctor explained them in simple language. Excellent service.",
            "I got an SMS when my medication was ready which saved me a lot of time. Very convenient.",
            "The midwives were caring and patient with me. The ward was spotless and the beds were comfortable.",
            "I arrived on time but was kept waiting for over an hour. The staff did not apologise or give any update.",
            "My appointment was cancelled without any notice. The receptionist was rude when I asked for help.",
            "We were seen within twenty minutes which was impressive. The doctor listened carefully.",
            "The dentist was gentle and made me feel at ease. The clinic was modern and very clean.",
            "The pharmacy staff were rude and dismissive. I waited an hour just to be told my medication was out of stock.",
            "The doctor explained each step and answered all my questions. I felt supported throughout.",
        ],
    }
)


MAX_ROWS = 20000  # very large files are sampled so the app stays fast

PRODUCT_SAMPLE = pd.DataFrame(
    {
        "review_date": ["2026-01-05", "2026-01-09", "2026-01-14", "2026-01-19", "2026-01-23", "2026-02-02",
                        "2026-02-06", "2026-02-11", "2026-02-16", "2026-02-21", "2026-02-26", "2026-03-03"],
        "product": ["Headphones", "Headphones", "Blender", "Blender", "Backpack", "Backpack",
                    "Headphones", "Blender", "Backpack", "Headphones", "Blender", "Backpack"],
        "stars": [5, 2, 4, 1, 5, 3, 4, 5, 2, 5, 2, 4],
        "review": [
            "Sound quality is fantastic and the battery lasts all week. Totally worth the price.",
            "The left earcup stopped working after a month. Support was slow to reply and the refund took forever.",
            "Powerful motor and easy to clean. A bit loud, but it crushes ice perfectly.",
            "Arrived with a cracked jug. Delivery was late and the packaging was terrible.",
            "Comfortable straps and plenty of space. Looks great and feels sturdy.",
            "Nice design but the zip feels cheap. Delivery was quick though.",
            "Comfortable fit and good noise cancelling. The app is confusing to set up.",
            "Best purchase this year. Smooth smoothies every morning and great value.",
            "The strap ripped within two weeks. Customer service was unhelpful and rude.",
            "Delivery was fast and the packaging was lovely. Sound is crisp and clear.",
            "Stopped working after three uses. Poor quality for the price.",
            "Great size for travel and very light. Support answered my question quickly.",
        ],
    }
)

STAFF_SAMPLE = pd.DataFrame(
    {
        "submitted": ["2026-02-03", "2026-02-03", "2026-02-04", "2026-02-05", "2026-02-06",
                      "2026-02-07", "2026-02-10", "2026-02-11", "2026-02-12", "2026-02-13"],
        "team": ["Engineering", "Sales", "Support", "Engineering", "Operations",
                 "Sales", "Support", "Operations", "Engineering", "Sales"],
        "what_is_working_well": [
            "Supportive manager and flexible hours.", "Great team spirit and good commission.",
            "I love helping customers and my colleagues are kind.", "Interesting projects and room to learn.",
            "Clear processes and a safe workplace.", "Targets are fair and leadership is open.",
            "Good training for new starters.", "Friendly colleagues.", "Great learning culture.",
            "Supportive and helpful team.",
        ],
        "what_should_improve": [
            "Too many meetings and unclear priorities.", "Tools are slow and training is limited.",
            "Workload is heavy and breaks are too short.", "Communication between teams is poor.",
            "Management rarely listens to feedback.", "The office is noisy and the equipment is outdated.",
            "Pay is low compared with the workload.", "Shift changes happen with no notice, which is stressful.",
            "Career growth paths are unclear.", "Too much pressure at month end.",
        ],
    }
)

SAMPLE_DATASETS = {
    "Patient feedback (healthcare)": SAMPLE_DATA,
    "Product reviews (with star ratings)": PRODUCT_SAMPLE,
    "Staff survey (two text columns)": STAFF_SAMPLE,
}

AUTO_THEMES = "Auto-discover from my data"
CUSTOM_THEMES = "Write my own"
CUSTOM_TEMPLATE = "Theme name: keyword, keyword, keyword\nAnother theme: keyword, keyword"

THEME_PRESETS = {
    "Healthcare / patient feedback": DEFAULT_THEMES,
    "Product & e-commerce reviews": """Quality: quality, build, sturdy, durable, cheap, broke, stopped working, material
Price and Value: price, value, worth, expensive, afford
Delivery and Packaging: deliver, shipping, arrived, package, packaging, late, courier
Customer Support: support, service, refund, replace, return, agent
Ease of Use: easy, simple, setup, set up, confusing, instructions, app
Design and Comfort: design, look, comfort, size, fit, light, heavy""",
    "Restaurants and hospitality": """Food: food, taste, flavour, meal, dish, menu, fresh, portion
Service: service, waiter, waitress, staff, friendly, rude, server
Ambience: ambience, atmosphere, music, noise, decor, cosy, seating
Cleanliness: clean, dirty, hygiene, toilet, table
Price and Value: price, value, expensive, worth, bill
Waiting and Booking: wait, queue, booking, reservation, slow, delay""",
    "Customer support": """Response Time: response, reply, wait, slow, quick, fast, hours, days
Resolution: resolve, solved, fixed, issue, problem, refund
Agent Behaviour: agent, rep, polite, rude, helpful, friendly, patient
Communication: explain, update, communicat, clear, confusing, follow up
Process: process, form, transfer, escalat, ticket, verify""",
    "Education / course feedback": """Content: content, material, topic, syllabus, relevant, examples
Instructor: instructor, lecturer, teacher, tutor, explain
Pace and Workload: pace, workload, fast, slow, assignment, deadline
Assessment: assessment, exam, test, grade, feedback, marking
Platform and Tools: platform, video, website, app, download, technical
Support: support, help, response, admin, question""",
    "Employee / workplace": """Management: manager, management, leadership, leader, boss
Workload and Wellbeing: workload, stress, pressure, hours, breaks, burnout, overtime
Pay and Benefits: pay, salary, bonus, benefits, commission
Culture and Team: team, colleague, culture, supportive, friendly, kind
Growth and Learning: career, growth, training, learn, promotion, development
Tools and Environment: tools, equipment, office, noisy, system, outdated
Communication: communicat, meetings, priorities, listens, feedback""",
}
SAMPLE_PRESET = {
    "Patient feedback (healthcare)": "Healthcare / patient feedback",
    "Product reviews (with star ratings)": "Product & e-commerce reviews",
    "Staff survey (two text columns)": "Employee / workplace",
}
THEME_STOPWORDS = STOPWORDS | set(
    """good great bad nice excellent terrible awful love loved hate amazing really very quite well best worst
    better worse always never still even also would could should thing things like lot much got went make
    made said came been need needs want wanted take took give gave back give know feel felt find found""".split()
)
RATING_HINTS = ["rating", "stars", "star", "score", "nps", "satisfaction", "grade", "csat"]


# ----------------------------------------------------------------------
# HELPER FUNCTIONS
# ----------------------------------------------------------------------
@st.cache_resource
def get_analyzer():
    """Create the VADER sentiment analyzer once and reuse it."""
    return SentimentIntensityAnalyzer()


def label_sentiment(score):
    """Turn a VADER compound score (-1 to +1) into a label."""
    if score >= POSITIVE_CUTOFF:
        return "Positive"
    if score <= NEGATIVE_CUTOFF:
        return "Negative"
    return "Neutral"


@st.cache_data
def score_texts(texts):
    """Return the VADER compound score for each text in the list."""
    analyzer = get_analyzer()
    return [analyzer.polarity_scores(t)["compound"] for t in texts]


def guess_column(df, hints, exclude):
    """Find a column whose name contains one of the hint words."""
    for col in df.columns:
        if col != exclude and any(h in str(col).lower() for h in hints):
            return col
    return None


def load_csv(file):
    """Read almost any CSV/TSV/TXT export: guesses encoding and separator, skips broken lines,
    and copes with files that have no header row or only one column of text."""
    raw = file.getvalue() if hasattr(file, "getvalue") else file.read()
    text = None
    for encoding in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue

    def parse(sep_kwargs, **extra):
        return pd.read_csv(io.StringIO(text), on_bad_lines="skip", **sep_kwargs, **extra)

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    try:  # only commas, semicolons, tabs and pipes count as separators (never spaces)
        delimiter = csv.Sniffer().sniff("\n".join(lines[:50]), delimiters=",;\t|").delimiter
    except Exception:
        delimiter = None
    if delimiter is None and lines:  # sniffing failed: does the first (header-like) line contain a separator?
        for candidate in (",", ";", "\t", "|"):
            if candidate in lines[0] and len(lines[0]) <= 150:
                delimiter = candidate
                break
    if delimiter is None:  # no table structure: treat every line as one comment
        return pd.DataFrame({"text": lines})
    used = {"sep": delimiter}
    try:
        df = parse(used)
    except Exception:
        return pd.DataFrame({"text": lines})

    # A very long "column name" means the first row was really a comment: reload without a header
    if any(len(str(c)) > 60 for c in df.columns):
        df = parse(used, header=None)
        df.columns = [f"column_{i + 1}" for i in range(len(df.columns))]

    df.columns = [str(c).strip() for c in df.columns]
    skipped_lines = max(0, len(lines) - 1 - len(df))  # lines pandas could not fit into the table
    df = df.dropna(how="all").dropna(axis=1, how="all").reset_index(drop=True)
    df.attrs["skipped_lines"] = skipped_lines
    return df


def text_columns(df):
    """Columns that hold free text, longest average text first."""
    found = []
    for col in df.columns:
        if df[col].dtype == object or str(df[col].dtype).startswith("str"):
            avg_len = df[col].dropna().astype(str).str.len().mean()
            if avg_len and avg_len >= 8:
                found.append((avg_len, col))
    return [col for _, col in sorted(found, reverse=True)]


def default_text_columns(df):
    """The longest text column, plus any other long free-text columns (for example survey questions)."""
    cols = text_columns(df)
    if not cols:
        return []
    long_ones = [c for c in cols if df[c].dropna().astype(str).str.len().mean() >= 30]
    return long_ones[:3] if len(long_ones) > 1 else cols[:1]


def combine_text(df, cols):
    """Join several text columns into one so each respondent is scored on everything they wrote."""
    if len(cols) == 1:
        return df, cols[0]
    out = df.copy()
    name = "combined_text"
    out[name] = out[cols].apply(
        lambda row: ". ".join(
            str(v).strip() for v in row if pd.notna(v) and str(v).strip() and str(v).lower() != "nan"
        ),
        axis=1,
    )
    return out, name


def is_date_like(df, col):
    """True for columns that hold dates (by name or because most values parse as dates)."""
    if any(h in str(col).lower() for h in ("date", "time", "month", "submitted", "created")):
        return True
    if df[col].dtype == object or str(df[col].dtype).startswith("str"):
        sample = df[col].dropna().astype(str).head(200)
        if sample.empty:
            return False
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return pd.to_datetime(sample, errors="coerce").notna().mean() >= 0.9
    return False


def rating_candidates(df, exclude):
    """Numeric columns with a handful of distinct values, such as 1-5 stars or a 0-10 score."""
    found = []
    for col in df.columns:
        if col in exclude:
            continue
        numbers = pd.to_numeric(df[col], errors="coerce")
        if numbers.notna().mean() >= 0.9 and 2 <= numbers.nunique() <= 12 and numbers.nunique() < numbers.notna().sum():
            found.append(col)
    return found


def looks_non_english(texts):
    """VADER is built for English, so warn when most letters are not plain English letters."""
    letters = [ch for ch in " ".join(list(texts)[:300]) if ch.isalpha()]
    return len(letters) >= 50 and sum(ch.isascii() for ch in letters) / len(letters) < 0.85


def auto_themes(texts, n=8):
    """Discover themes from the data: the words that appear in the most different comments."""
    docs = [{w for w in re.findall(r"[a-z]{4,}", str(t).lower()) if w not in THEME_STOPWORDS} for t in texts]
    total = max(len(docs), 1)
    doc_freq = Counter(w for words in docs for w in words)
    minimum = max(2, round(0.02 * total))
    chosen = {}
    for word, count in doc_freq.most_common():
        if count < minimum or count > 0.6 * total:
            continue
        if any(word.startswith(k) or k.startswith(word) for k in chosen):
            continue  # skip near-duplicates such as doctor / doctors
        chosen[word] = [word]
        if len(chosen) == n:
            break
    return {w.title(): kws for w, kws in chosen.items()}


def rating_table(data, rating_col):
    """Average sentiment for each rating value (for example each star level)."""
    numbers = pd.to_numeric(data[rating_col], errors="coerce")
    tmp = data.assign(_rating=numbers).dropna(subset=["_rating"])
    if len(tmp) < 5 or tmp["_rating"].nunique() < 2:
        return None
    table = tmp.groupby("_rating").agg(
        average_score=("sentiment_score", "mean"), comments=("sentiment_score", "count")
    )
    table.index = [str(int(v)) if float(v).is_integer() else str(v) for v in table.index]
    return table.round(3)


def prepare_scored_data(df, text_col):
    """Clean the text column, score every row and add the results as new columns."""
    data = df.dropna(subset=[text_col]).copy()
    data[text_col] = data[text_col].astype(str).str.strip()
    data = data[~data[text_col].str.lower().isin(["", "nan", "none"])].reset_index(drop=True)
    data["sentiment_score"] = score_texts(tuple(data[text_col]))
    data["sentiment"] = data["sentiment_score"].apply(label_sentiment)
    return data


def parse_themes(text):
    """Turn the 'Theme: word, word' text box into a dictionary."""
    themes = {}
    for line in text.splitlines():
        if ":" in line:
            name, words = line.split(":", 1)
            keywords = [w.strip().lower() for w in words.split(",") if w.strip()]
            if name.strip() and keywords:
                themes[name.strip()] = keywords
    return themes


def split_into_parts(comment):
    """Split a comment into sentences, and again at 'but' / 'though' / 'however'."""
    sentences = re.split(r"(?<=[.!?])\s+", str(comment).strip())
    parts = []
    for sentence in sentences:
        pieces = re.split(r",?\s+(?:but|though|however)\s+", sentence)
        parts.extend(p.strip() for p in pieces if p.strip())
    return parts


def build_theme_rows(data, text_col, themes):
    """Score each sentence separately and tag it with the themes it mentions."""
    rows = []
    analyzer = get_analyzer()
    for idx, text in data[text_col].items():
        for part in split_into_parts(text):
            # "waiting room" is about facilities, not waiting time, so rename it for matching
            match_text = part.lower().replace("waiting room", "lobby room").replace(
                "waiting area", "lobby area"
            )
            matched = [
                name
                for name, keywords in themes.items()
                if any(re.search(r"\b" + re.escape(k), match_text) for k in keywords)
            ]
            if matched:
                score = analyzer.polarity_scores(part)["compound"]
                for name in matched:
                    rows.append(
                        {
                            "row": idx,
                            "theme": name,
                            "text": part,
                            "sentiment_score": score,
                            "sentiment": label_sentiment(score),
                        }
                    )
    return pd.DataFrame(rows)


def summarise(df, by):
    """Average score and sentiment percentages for each value in column `by`."""
    summary = df.groupby(by).agg(
        mentions=("sentiment_score", "count"),
        average_score=("sentiment_score", "mean"),
        percent_positive=("sentiment", lambda s: (s == "Positive").mean() * 100),
        percent_neutral=("sentiment", lambda s: (s == "Neutral").mean() * 100),
        percent_negative=("sentiment", lambda s: (s == "Negative").mean() * 100),
    )
    return summary.round(2).sort_values("average_score", ascending=False)


def mix_table(summary):
    """Percentages in the order Positive, Neutral, Negative (for stacked charts)."""
    mix = summary[["percent_positive", "percent_neutral", "percent_negative"]].copy()
    mix.columns = ORDER
    return mix


def top_words(texts, n=8):
    """Most common meaningful words in a collection of texts."""
    counter = Counter()
    for text in texts:
        words = re.findall(r"[a-z']{3,}", str(text).lower())
        counter.update(w for w in words if w not in STOPWORDS)
    return counter.most_common(n)


def trend_table(data, date_col):
    """Average sentiment per day / week / month, depending on how long the data spans."""
    dates = pd.to_datetime(data[date_col], errors="coerce")
    tmp = data.assign(_date=dates).dropna(subset=["_date"])
    if len(tmp) < 5:
        return None
    span_days = (tmp["_date"].max() - tmp["_date"].min()).days
    freq = "D" if span_days <= 14 else "W" if span_days <= 120 else "M"
    tmp["period"] = tmp["_date"].dt.to_period(freq)
    table = tmp.groupby("period").agg(
        average_score=("sentiment_score", "mean"), comments=("sentiment_score", "count")
    )
    # Label each period by its start date (e.g. 2026-03-02, or 2026-03 for months)
    date_format = "%Y-%m" if freq == "M" else "%Y-%m-%d"
    table.index = table.index.to_timestamp().strftime(date_format)
    return table.round(3)


def make_insights(data, text_col, theme_df, group_col, date_col, rating_col=None):
    """Write the findings as plain-English sentences."""
    insights = []
    total = len(data)
    pos = (data["sentiment"] == "Positive").mean() * 100
    neu = (data["sentiment"] == "Neutral").mean() * 100
    neg = (data["sentiment"] == "Negative").mean() * 100
    avg = data["sentiment_score"].mean()

    if pos - neg > 15:
        mood = "leaning positive"
    elif neg - pos > 15:
        mood = "leaning negative"
    else:
        mood = "mixed"
    insights.append(
        f"Across {total:,} comments, {pos:.0f}% are positive, {neu:.0f}% neutral and "
        f"{neg:.0f}% negative (average score {avg:+.2f}), so overall sentiment is {mood}."
    )
    if neu >= 40:
        insights.append(
            f"{neu:.0f}% of comments scored as neutral. Factual complaints such as "
            "'I waited two hours' often score as neutral, so negative feeling may be understated."
        )

    # Themes
    if theme_df is not None and not theme_df.empty:
        ts = summarise(theme_df, "theme")
        ts = ts[ts["mentions"] >= MIN_MENTIONS]
        if len(ts) >= 2:
            best, worst = ts.index[0], ts.index[-1]
            insights.append(
                f"Strongest theme: {best} (average {ts.loc[best, 'average_score']:+.2f}, "
                f"{ts.loc[best, 'percent_positive']:.0f}% positive). "
                f"Weakest theme: {worst} (average {ts.loc[worst, 'average_score']:+.2f}, "
                f"{ts.loc[worst, 'percent_negative']:.0f}% negative)."
            )
        for name, row in ts.iterrows():
            if row["percent_positive"] >= 35 and row["percent_negative"] >= 35:
                insights.append(
                    f"{name} is polarised: {row['percent_positive']:.0f}% positive but "
                    f"{row['percent_negative']:.0f}% negative, so the average hides strong opposing views."
                )

    # Groups
    if group_col:
        gs = summarise(data, group_col)
        gs = gs[gs["mentions"] >= MIN_MENTIONS]
        if len(gs) >= 2:
            best, worst = gs.index[0], gs.index[-1]
            insights.append(
                f"By {group_col}: '{best}' scores highest ({gs.loc[best, 'average_score']:+.2f}) "
                f"and '{worst}' scores lowest ({gs.loc[worst, 'average_score']:+.2f})."
            )

    # Trend
    if date_col:
        trend = trend_table(data, date_col)
        if trend is not None and len(trend) >= 2:
            first, last = trend.index[0], trend.index[-1]
            change = trend["average_score"].iloc[-1] - trend["average_score"].iloc[0]
            if abs(change) < 0.05:
                direction = "stayed about the same"
            elif change > 0:
                direction = "improved"
            else:
                direction = "declined"
            insights.append(
                f"Sentiment has {direction} over time (from {trend['average_score'].iloc[0]:+.2f} "
                f"in {first} to {trend['average_score'].iloc[-1]:+.2f} in {last})."
            )

    # Ratings vs written sentiment
    if rating_col:
        r = pd.to_numeric(data[rating_col], errors="coerce")
        ok = r.notna()
        if ok.sum() >= 10 and r[ok].nunique() >= 2:
            corr = data.loc[ok, "sentiment_score"].corr(r[ok])
            if pd.notna(corr):
                how = "agree strongly" if corr >= 0.5 else "agree moderately" if corr >= 0.25 else "only weakly agree"
                insights.append(f"Ratings and written sentiment {how} (correlation {corr:+.2f}).")
                high_but_negative = int(((r >= r[ok].max() - 1) & (data["sentiment"] == "Negative")).sum())
                low_but_positive = int(((r <= r[ok].min() + 1) & (data["sentiment"] == "Positive")).sum())
                if high_but_negative or low_but_positive:
                    insights.append(
                        f"{high_but_negative} highly rated comments read as negative and {low_but_positive} "
                        "low-rated comments read as positive. These mismatches are worth reading."
                    )

    # Words
    neg_words = top_words(data.loc[data["sentiment"] == "Negative", text_col], 5)
    if neg_words:
        insights.append(
            "Most common words in negative comments: " + ", ".join(w for w, _ in neg_words) + "."
        )
    pos_words = top_words(data.loc[data["sentiment"] == "Positive", text_col], 5)
    if pos_words:
        insights.append(
            "Most common words in positive comments: " + ", ".join(w for w, _ in pos_words) + "."
        )

    insights.append(
        "Note: this tool uses a rule-based scorer (VADER). It can miss sarcasm and context, "
        "so treat the results as a guide and read some comments yourself."
    )
    return insights


# ----------------------------------------------------------------------
# LOOK & FEEL: editorial, stippled, dark-teal. Art is decorative; every
# real chart sits on a solid black panel.
# ----------------------------------------------------------------------
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,400;12..96,500;12..96,700&family=Space+Mono&family=DM+Sans:wght@400;500&display=swap');

html, body, .stApp, .stMarkdown, .stMarkdown p, label, input, textarea, button { font-family: 'DM Sans', sans-serif; }
h1, h2, h3, h4 { font-family: 'Bricolage Grotesque', sans-serif !important; font-weight: 500 !important; letter-spacing: -0.02em; }
.stApp { background: #0A2230; color: #E9F1EC; }
#MainMenu, footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }
.block-container, [data-testid="stMainBlockContainer"] { padding: 0 !important; max-width: 100% !important; }
.st-key-body { padding: 0 6vw 40px 6vw; }
section[data-testid="stSidebar"] { background: #071A25; border-right: 1px solid #16394a; }

/* Hero (the artwork image is injected separately) */
.hero { position: relative; width: 100%; aspect-ratio: 16 / 9; background-size: 100% 100%; background-repeat: no-repeat; background-color: #DCEBE5; }
.hero-inner { position: absolute; left: 6vw; top: 17%; width: 44%; }
.hero h1 { font-size: clamp(2.3rem, 5.2vw, 4.8rem); line-height: 1.02; color: #0B2B3A; margin: 0 0 22px 0; }
.hero p { font-size: clamp(0.95rem, 1.25vw, 1.15rem); color: #3b5762; line-height: 1.6; max-width: 520px; margin: 0 0 28px 0; }
.cta { display: inline-block; padding: 11px 22px; border: 2px dotted #F2A31B; border-radius: 6px;
       font-family: 'Space Mono', monospace; font-size: 12px; letter-spacing: 0.18em; color: #0B2B3A; }
.chips { position: absolute; left: 6vw; bottom: 9%; display: flex; gap: 10px; flex-wrap: wrap; }
.chips span { font-family: 'Space Mono', monospace; font-size: 11px; letter-spacing: 0.16em; color: #F6EFC9;
              border: 1px solid #3c6a73; background: rgba(10,34,48,0.55); padding: 7px 12px; border-radius: 999px; }
@media (max-width: 900px) {
  .hero { aspect-ratio: auto; min-height: 92vh; background-size: cover; background-position: 70% 0; }
  .hero-inner { position: relative; left: 0; top: 0; width: auto; padding: 14vh 6vw 0 6vw; }
  .chips { position: relative; left: 0; bottom: 0; padding: 20px 6vw; }
}

/* Sections */
.sec { border-top: 1px solid #16394a; padding: 46px 0 18px 0; margin-top: 26px; }
.eyebrow { font-family: 'Space Mono', monospace; font-size: 11px; letter-spacing: 0.5em; text-transform: uppercase; color: #7FA6A0; }
.sec h2 { font-size: clamp(1.9rem, 3.6vw, 3rem); color: #EEF3EA; margin: 14px 0 10px 0; font-weight: 400 !important; }
.sec p { color: #8FA9AE; max-width: 640px; line-height: 1.6; margin: 0; }

/* Stat strip */
.stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); border: 1px solid #16394a; margin: 18px 0 26px 0; }
.stat { padding: 18px 22px; border-right: 1px solid #16394a; }
.stat span { font-family: 'Space Mono', monospace; font-size: 11px; letter-spacing: 0.2em; color: #7FA6A0; text-transform: uppercase; }
.stat b { display: block; font-family: 'Bricolage Grotesque', sans-serif; font-weight: 500; font-size: 42px; color: #F6EFC9; }

/* Dotted mood spectrum */
.spectrum { display: flex; justify-content: space-between; align-items: center; height: 40px; margin-top: 8px; }
.spectrum i { display: block; width: 7px; height: 7px; border-radius: 50%; }
.spectrum i.on { width: 22px; height: 22px; box-shadow: 0 0 0 4px #0A2230, 0 0 0 5px #F6EFC9; }
.slabels { display: flex; justify-content: space-between; font-family: 'Space Mono', monospace; font-size: 11px; letter-spacing: 0.14em; color: #7FA6A0; margin-bottom: 26px; }

/* Insights as an editorial list */
.ins { display: flex; gap: 20px; padding: 16px 0; border-bottom: 1px solid #16394a; color: #DDE8E3; line-height: 1.55; }
.ins .n { font-family: 'Space Mono', monospace; color: #F2A31B; padding-top: 2px; }

/* Real charts: always on solid black */
[class*="st-key-chart_"] { background: #000000; border: 1px solid #1d3b4a !important; border-radius: 14px; padding: 12px 16px; }
.ptitle { font-family: 'Space Mono', monospace; font-size: 11px; letter-spacing: 0.25em; text-transform: uppercase; color: #7FA6A0; margin-bottom: 6px; }

.step { padding: 22px; border: 1px solid #16394a; height: 100%; }
.step .num { font-family: 'Space Mono', monospace; color: #F2A31B; letter-spacing: 0.2em; }
.step h4 { margin: 8px 0 6px 0; color: #F6EFC9; }
.step p { color: #8FA9AE; margin: 0; font-size: 15px; }
.filenote { font-family: 'Space Mono', monospace; font-size: 11px; letter-spacing: 0.16em; color: #7FA6A0; padding: 18px 0 6px 0; }
.footer { text-align: center; color: #5f8189; font-family: 'Space Mono', monospace; font-size: 11px; letter-spacing: 0.2em; margin-top: 44px; }
</style>
"""


def inject_style():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def hero(art_b64, stats=None):
    """Full-width hero: stippled artwork as the background, text on top."""
    st.markdown(
        f"<style>.hero{{background-image:url(data:image/jpeg;base64,{art_b64});}}</style>"
        "<div class='hero'><div class='hero-inner'>"
        "<h1>Feedback<br>has a shape.</h1>"
        "<p>SentimentScope turns written comments into a landscape of feelings, "
        "then reads it back to you in plain English.</p>"
        "<span class='cta'>SCROLL TO EXPLORE &darr;</span></div>"
        + (f"<div class='chips'>{''.join(f'<span>{escape(c)}</span>' for c in stats)}</div>" if stats else "")
        + "</div>",
        unsafe_allow_html=True,
    )


def section(label, title, blurb=""):
    st.markdown(
        f"<div class='sec'><div class='eyebrow'>{escape(label)}</div><h2>{escape(title)}</h2>"
        f"<p>{escape(blurb)}</p></div>",
        unsafe_allow_html=True,
    )


def show_insights(lines):
    for i, line in enumerate(lines, 1):
        st.markdown(
            f"<div class='ins'><span class='n'>{i:02d}</span><div>{escape(line)}</div></div>",
            unsafe_allow_html=True,
        )


def lerp_hex(a, b, t):
    ca = [int(a[i:i + 2], 16) for i in (1, 3, 5)]
    cb = [int(b[i:i + 2], 16) for i in (1, 3, 5)]
    return "#%02x%02x%02x" % tuple(round(x + (y - x) * t) for x, y in zip(ca, cb))


def mood_spectrum(avg, n=41):
    """A row of dots from orange (negative) to teal (positive); the big ring marks the average."""
    marker = round((max(-1, min(1, avg)) + 1) / 2 * (n - 1))
    dots = []
    for i in range(n):
        t = i / (n - 1)
        colour = lerp_hex(NEG, NEU, t * 2) if t < 0.5 else lerp_hex(NEU, POS, (t - 0.5) * 2)
        dots.append(f"<i class='{'on' if i == marker else ''}' style='background:{colour}'></i>")
    st.markdown(
        f"<div class='spectrum'>{''.join(dots)}</div>"
        "<div class='slabels'><span>NEGATIVE</span><span>NEUTRAL</span><span>POSITIVE</span></div>",
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------------
# CHARTS (Altair) shown on black panels
# ----------------------------------------------------------------------
def style_chart(chart, height=300):
    return (
        chart.properties(width="container", height=height, background="transparent")
        .configure_axis(
            labelColor=MUTED, titleColor=MUTED, gridColor="#10252f",
            domainColor="#1d3b4a", tickColor="#1d3b4a", labelFont="DM Sans", titleFont="DM Sans",
        )
        .configure_legend(labelColor=MUTED, titleColor=MUTED, labelFont="DM Sans")
        .configure_view(strokeWidth=0)
    )


def chart_panel(chart, key, title):
    """Every real chart goes on a solid black panel so it can't be mistaken for the art."""
    with st.container(border=True, key=f"chart_{key}"):
        st.markdown(f"<div class='ptitle'>{escape(title)}</div>", unsafe_allow_html=True)
        st.altair_chart(chart, theme=None)


def donut_chart(counts):
    df = counts.rename_axis("sentiment").reset_index(name="count")
    arc = alt.Chart(df).mark_arc(innerRadius=75, outerRadius=125, padAngle=0.02).encode(
        theta=alt.Theta("count:Q"),
        color=alt.Color("sentiment:N", scale=SENT_SCALE, legend=alt.Legend(title=None, orient="bottom")),
        tooltip=["sentiment:N", "count:Q"],
    )
    return style_chart(arc, 320)


def score_bar_chart(series):
    df = series.rename_axis("name").reset_index(name="average_score")
    bars = alt.Chart(df).mark_bar(size=18).encode(
        x=alt.X("average_score:Q", title="Average sentiment score"),
        y=alt.Y("name:N", sort="-x", title=None),
        color=alt.condition("datum.average_score >= 0", alt.value(POS), alt.value(NEG)),
        tooltip=["name:N", alt.Tooltip("average_score:Q", format="+.2f")],
    )
    zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color="#F6EFC9", opacity=0.6).encode(x="x:Q")
    return style_chart(alt.layer(bars, zero), max(180, 46 * len(df) + 40))


def mix_chart(mix):
    df = mix.rename_axis("name").reset_index().melt(id_vars="name", var_name="sentiment", value_name="percent")
    df["order"] = df["sentiment"].map({s: i for i, s in enumerate(ORDER)})
    chart = alt.Chart(df).mark_bar(size=18).encode(
        x=alt.X("percent:Q", title="Share of mentions (%)", scale=alt.Scale(domain=[0, 100])),
        y=alt.Y("name:N", sort=list(mix.index), title=None),
        color=alt.Color("sentiment:N", scale=SENT_SCALE, legend=alt.Legend(title=None, orient="bottom")),
        order=alt.Order("order:Q"),
        tooltip=["name:N", "sentiment:N", alt.Tooltip("percent:Q", format=".0f")],
    )
    return style_chart(chart, max(180, 46 * len(mix) + 70))


def trend_chart(trend):
    df = trend.rename_axis("period").reset_index()
    base = alt.Chart(df).encode(x=alt.X("period:O", sort=None, title=None))
    area = base.mark_area(opacity=0.18, color=ACCENT, interpolate="monotone").encode(y="average_score:Q")
    line = base.mark_line(color=ACCENT, strokeWidth=3, interpolate="monotone",
                          point=alt.OverlayMarkDef(size=80, filled=True)).encode(
        y=alt.Y("average_score:Q", title="Average sentiment score"),
        tooltip=["period:O", alt.Tooltip("average_score:Q", format="+.2f"), "comments:Q"],
    )
    zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(strokeDash=[5, 5], color="#F6EFC9", opacity=0.5).encode(y="y:Q")
    return style_chart(alt.layer(zero, area, line), 320)


def word_chart(words, colour):
    df = pd.DataFrame(words, columns=["word", "count"])
    if df.empty:
        return None
    chart = alt.Chart(df).mark_bar(color=colour, size=14).encode(
        x=alt.X("count:Q", title=None),
        y=alt.Y("word:N", sort="-x", title=None),
        tooltip=["word:N", "count:Q"],
    )
    return style_chart(chart, 280)


@st.cache_data
def hero_art(scores, pos, neu, neg):
    """Cached so the artwork is only redrawn when the data changes."""
    return make_hero(list(scores), pos, neu, neg)


def rating_chart(table):
    df = table.rename_axis("rating").reset_index()
    bars = alt.Chart(df).mark_bar(size=34).encode(
        x=alt.X("rating:O", sort=list(df["rating"]), title="Rating"),
        y=alt.Y("average_score:Q", title="Average sentiment score"),
        color=alt.condition("datum.average_score >= 0", alt.value(POS), alt.value(NEG)),
        tooltip=["rating:O", alt.Tooltip("average_score:Q", format="+.2f"), "comments:Q"],
    )
    zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="#F6EFC9", opacity=0.6).encode(y="y:Q")
    return style_chart(alt.layer(bars, zero), 320)


# ----------------------------------------------------------------------
# THE APP
# ----------------------------------------------------------------------
def main():
    st.set_page_config(page_title="SentimentScope", page_icon="🔭", layout="wide")
    inject_style()

    # ---------------- Sidebar: load ----------------
    with st.sidebar:
        st.markdown("## SentimentScope")
        st.header("1. Load your data")
        uploaded = st.file_uploader("Upload a CSV file", type=["csv", "tsv", "txt"],
                                    help="Any export with a column of written comments. Separators and encodings are detected automatically.")
        use_sample = st.checkbox("Or try a built-in sample (fictional)")
        sample_name = st.selectbox("Sample dataset", list(SAMPLE_DATASETS)) if use_sample else None
        st.caption("Please do not upload data that identifies real people. Scoring is designed for English text.")

    df, problem, notes = None, None, []
    if uploaded is not None:
        try:
            df = load_csv(uploaded)
        except Exception as error:
            problem = f"That file couldn't be read ({error}). Try saving it again as a plain CSV."
    elif use_sample:
        df = SAMPLE_DATASETS[sample_name].copy()
    if df is not None and df.attrs.get("skipped_lines"):
        notes.append(f"{df.attrs['skipped_lines']} line(s) couldn't be read (for example, unquoted commas inside a comment) and were skipped.")
    if df is not None and problem is None and (df.empty or len(df.columns) == 0):
        problem = "The file looks empty. Please check it and try again."
    if df is not None and problem is None and len(df) > MAX_ROWS:
        notes.append(f"The file has {len(df):,} rows, so a random sample of {MAX_ROWS:,} is analysed to keep things fast.")
        df = df.sample(MAX_ROWS, random_state=0).reset_index(drop=True)
    rows_read, cols_read = (len(df), len(df.columns)) if df is not None and problem is None else (0, 0)

    # ---------------- Sidebar: choose columns and themes ----------------
    data = None
    if df is not None and problem is None:
        with st.sidebar:
            st.header("2. Choose columns")
            columns = list(df.columns)
            text_cols = st.multiselect("Column(s) with the text to analyse", columns,
                                       default=default_text_columns(df),
                                       help="Choose several to combine, for example two survey questions.")
            if not text_cols:
                problem = ("No text column is selected. Pick the column that holds the written comments "
                           f"(columns found: {', '.join(columns)}).")
            else:
                df, text_col = combine_text(df, text_cols)
                rest = [c for c in columns if c not in text_cols]
                group_options = ["(none)"] + [c for c in rest if 2 <= df[c].nunique() <= 50 and not is_date_like(df, c)]
                g_guess = guess_column(df, ["department", "category", "branch", "product", "location", "source", "segment", "group", "team", "region"], text_col)
                group_choice = st.selectbox("Compare by group (optional)", group_options,
                                            index=group_options.index(g_guess) if g_guess in group_options else 0)
                date_options = ["(none)"] + rest
                d_guess = guess_column(df, ["date", "time", "day", "month", "submitted", "created"], text_col)
                date_choice = st.selectbox("Date column for trends (optional)", date_options,
                                           index=date_options.index(d_guess) if d_guess in date_options else 0)
                rating_options = ["(none)"] + rating_candidates(df, set(text_cols))
                r_guess = next((c for c in rating_options[1:] if any(h in c.lower() for h in RATING_HINTS)), None)
                rating_choice = st.selectbox("Rating or score column (optional)", rating_options,
                                             index=rating_options.index(r_guess) if r_guess else 0,
                                             help="For example 1-5 stars. The tool compares ratings with the written sentiment.")
                group_col = None if group_choice == "(none)" else group_choice
                date_col = None if date_choice == "(none)" else date_choice
                rating_col = None if rating_choice == "(none)" else rating_choice

                st.header("3. Themes")
                preset_options = [AUTO_THEMES] + list(THEME_PRESETS) + [CUSTOM_THEMES]
                default_preset = SAMPLE_PRESET.get(sample_name, AUTO_THEMES) if uploaded is None else AUTO_THEMES
                preset = st.selectbox("Theme set", preset_options, index=preset_options.index(default_preset),
                                      help="Auto-discover finds the topics people mention most. Presets suit common domains.")
                theme_text = ""
                if preset != AUTO_THEMES:
                    theme_text = st.text_area("One theme per line: Name: keyword, keyword",
                                              THEME_PRESETS.get(preset, CUSTOM_TEMPLATE), height=240,
                                              key=f"themes_{preset}",
                                              help="Keywords match the start of words, so 'wait' also finds 'waiting'.")
                data = prepare_scored_data(df, text_col)
                if data.empty:
                    problem = "No readable text was found in the selected column(s). Please choose different columns."

    # ---------------- Hero (full width) ----------------
    if data is not None and problem is None:
        scores = data["sentiment_score"].to_numpy()
        pos = float((data["sentiment"] == "Positive").mean())
        neu = float((data["sentiment"] == "Neutral").mean())
        neg = float((data["sentiment"] == "Negative").mean())
        sample = scores if len(scores) <= 3000 else np.random.default_rng(0).choice(scores, 3000, replace=False)
        art = hero_art(tuple(np.round(sample, 3)), round(pos, 2), round(neu, 2), round(neg, 2))
        chips = [f"{len(data):,} VOICES", f"MOOD {scores.mean():+.2f}", f"{pos * 100:.0f}% POSITIVE", f"{neg * 100:.0f}% NEGATIVE"]
    else:
        demo = np.concatenate([np.random.default_rng(1).normal(0.45, 0.25, 300), np.random.default_rng(2).normal(-0.5, 0.25, 220)])
        art = hero_art(tuple(np.round(demo, 3)), 0.5, 0.15, 0.35)
        chips = None
    hero(art, chips)

    with st.container(key="body"):
        if problem:
            st.error(problem)
            if df is not None:
                with st.expander("Preview of the file"):
                    st.dataframe(df.head(15), hide_index=True)
            st.stop()
        if data is None:
            section("The method", "Three steps, then scroll", "Upload any feedback file, let the tool read it, then explore what it found.")
            c1, c2, c3 = st.columns(3)
            c1.markdown("<div class='step'><div class='num'>01</div><h4>Upload</h4><p>Any CSV, TSV or TXT with written comments: reviews, surveys, tickets, patient feedback. Optional columns for groups, dates and star ratings unlock more.</p></div>", unsafe_allow_html=True)
            c2.markdown("<div class='step'><div class='num'>02</div><h4>Read</h4><p>Every comment is scored Positive, Neutral or Negative, and themes are discovered automatically or taken from a preset.</p></div>", unsafe_allow_html=True)
            c3.markdown("<div class='step'><div class='num'>03</div><h4>Explore</h4><p>Scroll through charts, plain-English insights and the voices behind the numbers.</p></div>", unsafe_allow_html=True)
            st.info("Use the sidebar to upload a file, or tick the sample box to try one of three built-in datasets.")
            st.stop()

        # ---- what the tool understood about the file ----
        skipped = rows_read - len(data)
        st.markdown(
            f"<div class='filenote'>FILE &middot; {rows_read:,} ROWS &times; {cols_read} COLUMNS &nbsp;&middot;&nbsp; "
            f"{len(data):,} COMMENTS ANALYSED{f' ({skipped:,} EMPTY SKIPPED)' if skipped else ''} &nbsp;&middot;&nbsp; "
            f"TEXT FROM {escape(', '.join(text_cols).upper())}</div>",
            unsafe_allow_html=True,
        )
        if looks_non_english(data[text_col]):
            notes.append("Much of this text doesn't look like English. The scorer is built for English, so results may be unreliable.")
        if len(data) < 10:
            notes.append("There are fewer than 10 comments, so treat the results as a rough illustration.")
        for note in notes:
            st.warning(note)
        with st.expander("Preview your file"):
            st.dataframe(df.head(10), hide_index=True)

        themes = auto_themes(data[text_col]) if preset == AUTO_THEMES else parse_themes(theme_text)
        theme_df = build_theme_rows(data, text_col, themes) if themes else None
        insights = make_insights(data, text_col, theme_df, group_col, date_col, rating_col)
        numbers = iter(range(1, 20))

        def sec(label, title, blurb=""):
            section(f"{next(numbers):02d} / {label}", title, blurb)

        # ---- The reading ----
        sec("The reading", "How does everyone feel?", "The headline numbers, then what stands out.")
        st.markdown(
            "<div class='stats'>"
            f"<div class='stat'><span>Voices</span><b>{len(data):,}</b></div>"
            f"<div class='stat'><span>Mood</span><b>{scores.mean():+.2f}</b></div>"
            f"<div class='stat'><span>Positive</span><b>{pos * 100:.0f}%</b></div>"
            f"<div class='stat'><span>Negative</span><b>{neg * 100:.0f}%</b></div></div>",
            unsafe_allow_html=True,
        )
        mood_spectrum(scores.mean())
        left, right = st.columns([1, 1.2], gap="large")
        with left:
            chart_panel(donut_chart(data["sentiment"].value_counts().reindex(ORDER, fill_value=0)),
                        "donut", "Sentiment breakdown")
        with right:
            show_insights(insights)
        w1, w2 = st.columns(2, gap="large")
        for col, label, colour, key in [(w1, "Positive", POS, "wp"), (w2, "Negative", NEG, "wn")]:
            chart = word_chart(top_words(data.loc[data["sentiment"] == label, text_col], 10), colour)
            if chart is not None:
                with col:
                    chart_panel(chart, key, f"Common words in {label.lower()} comments")

        # ---- Themes ----
        blurb = ("These topics were discovered automatically from the words people use most. "
                 if preset == AUTO_THEMES else "") + "Each sentence is scored separately, so one comment can speak to several themes."
        sec("The themes", "What are people talking about?", blurb)
        if theme_df is None or theme_df.empty:
            st.info("No themes matched this data. Try 'Auto-discover', another theme set, or your own keywords in the sidebar.")
        else:
            ts = summarise(theme_df, "theme")
            a, b = st.columns(2, gap="large")
            with a:
                chart_panel(score_bar_chart(ts["average_score"]), "themescore", "Average sentiment by theme")
            with b:
                chart_panel(mix_chart(mix_table(ts)), "thememix", "Positive / neutral / negative mix")
            st.dataframe(ts)

        # ---- Groups ----
        if group_col:
            sec("The groups", f"How do the {group_col} values compare?", "Where feeling is strongest, and where it dips.")
            gs = summarise(data, group_col)
            if len(gs) > 15:
                gs = gs.sort_values("mentions", ascending=False).head(15).sort_values("average_score", ascending=False)
                st.caption("Showing the 15 groups with the most comments.")
            a, b = st.columns(2, gap="large")
            with a:
                chart_panel(score_bar_chart(gs["average_score"]), "groupscore", f"Average sentiment by {group_col}")
            with b:
                chart_panel(mix_chart(mix_table(gs)), "groupmix", "Positive / neutral / negative mix")
            st.dataframe(gs)

        # ---- Ratings ----
        rtable = rating_table(data, rating_col) if rating_col else None
        if rtable is not None:
            sec("The ratings", "Do the numbers match the words?", "Average written sentiment at each rating level.")
            chart_panel(rating_chart(rtable), "rating", f"Average sentiment by {rating_col}")

        # ---- Timeline ----
        trend = trend_table(data, date_col) if date_col else None
        if trend is not None:
            sec("The timeline", "Is feeling getting better or worse?", "Average sentiment over time.")
            chart_panel(trend_chart(trend), "trend", "Average sentiment over time")

        # ---- Voices ----
        sec("The voices", "Read the words behind the numbers", "The most negative and most positive comments, plus a search box.")
        show_cols = [c for c in [group_col, date_col, rating_col, text_col, "sentiment_score", "sentiment"] if c]
        show_cols = list(dict.fromkeys(show_cols))
        v1, v2 = st.columns(2, gap="large")
        with v1:
            st.caption("Most negative")
            st.dataframe(data.sort_values("sentiment_score").head(10)[show_cols], hide_index=True)
        with v2:
            st.caption("Most positive")
            st.dataframe(data.sort_values("sentiment_score", ascending=False).head(10)[show_cols], hide_index=True)
        query = st.text_input("Search all comments", placeholder="Type a word, for example: delivery")
        filtered = data[data[text_col].str.contains(query, case=False, na=False, regex=False)] if query else data
        st.dataframe(filtered[show_cols], hide_index=True)

        # ---- Download ----
        sec("Take it with you", "Download your results")
        d1, d2 = st.columns(2)
        d1.download_button("Download scored data (CSV)", data.to_csv(index=False).encode("utf-8"),
                           file_name="sentiment_results.csv", mime="text/csv")
        report = "SENTIMENT ANALYSIS INSIGHTS\n\n" + "\n".join(f"- {line}" for line in insights)
        d2.download_button("Download insights (TXT)", report.encode("utf-8"),
                           file_name="insights.txt", mime="text/plain")

        st.markdown("<div class='footer'>BUILT BY ELZEL DE BEER &nbsp;&middot;&nbsp; STIPPLED WITH PYTHON</div>",
                    unsafe_allow_html=True)


if __name__ == "__main__":
    main()
