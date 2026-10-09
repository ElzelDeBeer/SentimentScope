## INTRODUCTION: 

## PROJECT NAME
- SentimentScope

## BACKGROUND INFORMATION

SentimentScope is an interactive sentiment analysis and data insights tool. Upload a CSV file of comments (reviews, survey answers, patient feedback, support tickets) 
and it scores how people feel, finds what they are talking about, and writes the findings back in plain English. 
The page opens with generative stipple artwork drawn from your own data, then the analysis appears as you scroll.

## ACCESS THE TOOL USING THE LINK: 
- https://sentimentscope-8lssmwcjncg32pcxachmbm.streamlit.app/

## SENTIMENT SCOPE PREVIEW

- <img width="1598" height="786" alt="EQ DE BEER SA DI" src="https://github.com/user-attachments/assets/8726330c-300f-4181-bbef-52d3fc34e8fd" />
- <img width="1600" height="773" alt="EQ DE BEER SA DA" src="https://github.com/user-attachments/assets/7c4bb0a1-760f-4322-ae21-add54217b5ec" />
- <img width="1596" height="784" alt="EQ DE BEER SA   DI" src="https://github.com/user-attachments/assets/5047416c-98d6-4059-bad1-5ae76fc3ce8c" />

## What Sentiment Scope does: 

- Reads almost any text file: files such as CSV, TSV and TXT work on this tool. The separator and encoding are detected automatically,
  files without a header row are handled, and a plain list of comments (one per line) also works.
- Scores every comment as Positive, Neutral or Negative using VADER, with a score from -1 (very negative) to +1 (very positive).
- Finds themes: topics can be discovered automatically from the words people use most, taken from a preset (healthcare, product reviews, restaurants, customer support, education, workplace),
   or written by you. Each sentence is scored separately, so one comment can speak to several themes.
- Compares groups: examples include departments, products or teams, and shows trends over time when the file has a date column.
- Compares ratings with the words: if the file has a star or score column, the tool checks how well the ratings match the written sentiment and flags mismatches.
- Writes the insights in plain English, and lets you download the scored data (CSV) and the insights (TXT).
- Generates artwork from your data: the dotted hills show the distribution of your sentiment scores, and the circular figure is a radial histogram (orange for negative, teal for positive).
- Keeps art and data separate: every real chart sits on a solid black panel, so decorative graphics are never confused with data.
- Includes three fictional sample datasets such as patient feedback, product reviews with ratings, and a two-question staff survey so a user can try it without a file.

## How it works

- The file is loaded and cleaned and the empty rows and blank comments are skipped.
- Each comment is scored with VADER. So, scores of +0.05 or above are Positive, -0.05 or below are Negative, and anything in between is Neutral.
- Comments are split into sentences and at words like "but" and "though" so mixed opinions are scored separately, then tagged with themes by keyword matching.
- Results are summarised by theme, group, rating and date, and turned into plain-English insights.
- Charts are drawn with Altair, and the hero artwork is drawn with matplotlib and cached until the data changes.

## TECH STACK

 Programming Language Used - Python 3 
 Web app - [Streamlit](https://streamlit.io)
 Sentiment scoring - [VADER](https://github.com/cjhutto/vaderSentiment) (`vaderSentiment`) 
 Data handling - pandas, NumPy
 Charts - Altair 
 Generative artwork - matplotlib 
 Styling - Custom CSS, Google Fonts (Bricolage Grotesque, Space Mono, DM Sans) 
 Hosting - Streamlit Community Cloud 

## PROJECT STRUCTURE 

SentimentScope/
├── app.py              # The Streamlit app: loading, analysis, charts and layout
├── art.py              # Generates the stipple hero artwork from the data
├── requirements.txt    # Python libraries the app needs
└── .streamlit/
    └── config.toml     # Dark theme colours

## GETTING STARTED 

### 1. What you need

- Python 3.14 (64-bit) 
- An internet connection as the fonts load from Google Fonts

### 2. Get the code

git clone https://github.com/ElzelDeBeer/SentimentScope.git
cd SentimentScope

- No Git? On the GitHub page click 'Code', then 'Download ZIP', unzip it, and open a terminal inside the folder.
- Open the folder in File Explorer, click the address bar, type `cmd` and press Enter. A terminal opens in that folder.

### 3. Install the libraries, only install the libraries once

Use the first command. If it doesn't work, try the next one.

 py -m pip install -r requirements.txt -> Windows 
 python -m pip install -r requirements.txt -> If `py` is not recognized 
 python3 -m pip install -r requirements.txt -> Mac and Linux |
 pip install -r requirements.txt -> If none of the above work 

### 4. Run the app

Again, if one command doesn't work, try the next.

py -m streamlit run app.py -> Windows (this is the most common) 
python -m streamlit run app.py -> If `py` is not recognised 
python3 -m streamlit run app.py -> Mac and Linux 
streamlit run app.py -> If the ones above fail but Streamlit installed 

After running the commands your browser should open the app. If it doesn't, go to **http://localhost:8501**.
The first time, Streamlit may ask for an email address. You can press **Enter** to skip it.

### 5. Optional: You can use a virtual environment

This keeps the project's libraries separate from the rest of your computer.

# FOR Windows
py -m venv .venv
.venv\Scripts\activate

# FOR Mac and Linux
python3 -m venv .venv
source .venv/bin/activate

Then repeat steps 3 and 4. To leave the environment, type `deactivate`.

### TROUBLESHOOTING

Problem -> What to try 

'py' is not recognized` -> Use python or python3 instead, or reinstall Python and tick **Add Python to PATH** 
 No module named streamlit (or another library) -> Run the install command in step 3 again, using the same command prefix you use to run the app 
 No module named art -> `art.py` must be in the same folder as `app.py` 
 Port already in use -> Add a port: `py -m streamlit run app.py --server.port 8502`
 Page opens but looks light and unstyled -> Check that `config.toml` is inside a folder named `.streamlit` 
 Browser did not open -> Copy the `Local URL` shown in the terminal into your browser 
 Install fails on a very new Python version -> Try Python 3.12 or 3.13 


## USING THE TOOL

- In the sidebar, choose upload a file, or tick the sample box and choose a sample dataset.
- Check the text column the tool picked. Choose several if your file has more than one comment column.
- Optionally choose a group, a date and a rating column to unlock more sections.
- Pick a theme set, Nota Bene: auto-discover is the default for uploaded files.
- Scroll down to read the results, then download the scored data or the insights.

Your file needs: at least one column of written comments. Everything else is optional.

## PRIVACY

- The sample data in this project is entirely fictional.
- Please do not upload data that identifies real people, such as names, ID numbers or contact details.
- The app does not save uploaded files. They are processed in memory during your session.

## LIMITATIONS 

- **English only.** VADER is designed for English text. Other languages can give unreliable scores, and the app warns you when text doesn't look English.
- **Rule-based scoring.** VADER does not understand context. It can miss sarcasm, slang and medical or technical meaning, and factual complaints such as "I waited three hours" often score as neutral, so negative feeling can be understated.
- **Themes use keywords.** A sentence is tagged with a theme only if it contains one of the theme's keywords, so related wording can be missed. Auto-discovered themes are based on word frequency, not meaning.
- **Small files give rough results.** With only a few comments, themes and group comparisons are illustrative rather than reliable.
- **Large files are sampled.** Files over 20,000 rows are randomly sampled to keep the app fast.
- **File parsing has limits.** Badly broken files (for example unquoted commas inside comments) may lose some lines. The app tells you when lines are skipped.
- **Needs an internet connection** for the web fonts.
- **No automated tests yet.** The analysis logic has been checked by hand and with sample files, but there is no test suite.

## FUTURE IMPROVEMENTS 

- Using a modern language model for sentiment, for example a transformer-based model to handle context and sarcasm better
- Support multiple languages, including South African languages
- Add aspect-based sentiment, so each theme gets its own score within a comment
- Use topic modelling or embeddings for smarter automatic themes
- Detect emotions such as anger, joy, frustration, not just positive and negative
- Import Excel files and Google Sheets
- Export a ready-made PDF or Word report with the charts
- Compare two files side by side, for example before and after a change
- Let users adjust the sentiment thresholds and add custom stop words
- Add automated tests and a colour-blind-friendly palette option
- Add alerts when sentiment drops sharply over time

## DEPLOYMENT

The app is deployed on [Streamlit Community Cloud](https://streamlit.io/cloud). 

## AUTHOR

This tool was built by me, Elzel de Beer as a data insights portfolio project.

## ACKNOWLEDGEMENTS 

- Sentiment scoring by [VADER](https://github.com/cjhutto/vaderSentiment) (Hutto and Gilbert, 2014)
- App framework by [Streamlit](https://streamlit.io)
