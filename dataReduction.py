import csv
from datetime import datetime
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from collections import defaultdict

# Initialize VADER analyzer
analyzer = SentimentIntensityAnalyzer()

# Dictionary to accumulate per-day statistics
daily_stats = defaultdict(lambda: {
    "total": 0,
    "positive": 0,
    "neutral": 0,
    "negative": 0,
    "compound_sum": 0.0
})

# Read input line-by-line
with open("LotsOfTweets.csv", 'r', encoding='utf-8') as infile:
    reader = csv.DictReader(infile)
    total_lines = sum(1 for _ in open("LotsOfTweets.csv", encoding='utf-8')) - 1
    infile.seek(0)  # Reset file pointer after counting

    print(f"📊 Processing {total_lines} tweets...")

    for i, row in enumerate(reader, 1):
        try:
            date = datetime.strptime(row['Date'][:10], "%Y-%m-%d").date()
            text = row['text']
            score = analyzer.polarity_scores(text)
            compound = score['compound']

            # Classify sentiment
            if compound >= 0.05:
                sentiment = "positive"
            elif compound <= -0.05:
                sentiment = "negative"
            else:
                sentiment = "neutral"

            # Update stats
            stats = daily_stats[date]
            stats["total"] += 1
            stats[sentiment] += 1
            stats["compound_sum"] += compound

            # Print progress every 10,000 lines
            if i % 10000 == 0 or i == total_lines:
                print(f"Processed {i}/{total_lines} tweets ({(i / total_lines) * 100:.2f}%)")

        except Exception as e:
            print(f"❌ Error on line {i}: {e}")

# Write the daily summary line-by-line
with open("DailySentimentSummary.csv", 'w', newline='', encoding='utf-8') as outfile:
    writer = csv.writer(outfile)
    writer.writerow(['Date', 'Total_Tweets', 'Positive_Tweets', 'Neutral_Tweets', 'Negative_Tweets', 'Avg_Compound_Score'])

    for date in sorted(daily_stats.keys()):
        stats = daily_stats[date]
        avg_compound = stats['compound_sum'] / stats['total'] if stats['total'] else 0.0
        writer.writerow([
            date,
            stats['total'],
            stats['positive'],
            stats['neutral'],
            stats['negative'],
            round(avg_compound, 6)
        ])

print("✅ Summary saved to 'DailySentimentSummary.csv'")
