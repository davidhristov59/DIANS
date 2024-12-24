import pdfplumber
import csv
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def extract_text_from_pdf(pdf_file_path):
    text = ""
    try:
        with pdfplumber.open(pdf_file_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text()
    except Exception as e:
        print(f"Error reading {pdf_file_path}: {e}")
    return text

def perform_sentiment_analysis(text):
    analyzer = SentimentIntensityAnalyzer()
    sentiment_scores = analyzer.polarity_scores(text)
    return sentiment_scores

def make_stock_recommendation(sentiment_scores):
    compound_score = sentiment_scores['compound']

    if compound_score > 0.1:
        recommendation = "Buy"
        predicted_price_change = "increase"
    elif compound_score < -0.1:
        recommendation = "Sell"
        predicted_price_change = "decrease"
    else:
        recommendation = "Hold"
        predicted_price_change = "remain stable"

    return recommendation, predicted_price_change

def process_issuer(issuer_name):
    pdf_file_path = f'sentimentalAnalysis/{issuer_name}.pdf'

    print(f"Extracting text from {issuer_name}.pdf...")
    pdf_text = extract_text_from_pdf(pdf_file_path)

    if pdf_text:
        print(f"Performing sentiment analysis for {issuer_name}...")
        sentiment_scores = perform_sentiment_analysis(pdf_text)

        recommendation, predicted_price_change = make_stock_recommendation(sentiment_scores)

        description = (
            f"Sentiment Scores: {sentiment_scores}\n"
            f"Predicted stock price will {predicted_price_change}.\n"
            f"Recommendation: {recommendation} stocks."
        )

        result = {
            "issuer": issuer_name,
            "description": description
        }

        return result
    else:
        return {
            "issuer": issuer_name,
            "description": "Error in processing or no text extracted."
        }

def analyze_all_issuers(issuers, output_file):
    print(f"Starting fundamental analysis for {len(issuers)} issuers...")

    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.DictWriter(csvfile, fieldnames=["Issuer Name", "Description"])
        csv_writer.writeheader()

        for issuer in issuers:
            result = process_issuer(issuer)
            csv_writer.writerow({"Issuer Name": result['issuer'], "Description": result['description']})

    print(f"Results saved to {output_file}")

def main():
    issuers = ['ALK', 'CKB', 'GRNT', 'KMB', 'MPT', 'MSTIL', 'MTUR', 'REPL', 'STB', 'SBT', 'TEL', 'TTK', 'TNB', 'UNI', 'VITA']

    output_file = 'all_issuers_analysis.csv'

    analyze_all_issuers(issuers, output_file)

if __name__ == '__main__':
    main()
