import pdfplumber
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

        result = {
            "issuer": issuer_name,
            "sentiment_scores": sentiment_scores,
            "recommendation": recommendation,
            "predicted_price_change": predicted_price_change
        }

        return result
    else:
        return None

def analyze_all_issuers(issuers, output_file):
    print(f"Starting fundamental analysis for {len(issuers)} issuers...")

    with open(output_file, 'w') as f:
        f.write("Fundamental Analysis Results for All Issuers\n")
        f.write("=" * 50 + "\n\n")

        for issuer in issuers:
            result = process_issuer(issuer)
            if result:
                f.write(f"Issuer: {result['issuer']}\n")
                f.write(f"Sentiment Scores: {result['sentiment_scores']}\n")
                f.write(f"Predicted stock price will {result['predicted_price_change']}.\n")
                f.write(f"Recommendation: {result['recommendation']} stocks.\n")
                f.write("\n" + "-" * 50 + "\n\n")
            else:
                f.write(f"Issuer: {issuer} - Error in processing or no text extracted.\n")
                f.write("\n" + "-" * 50 + "\n\n")

    print(f"Results saved to {output_file}")

def main():
    issuers = ['ALK', 'CKB', 'GRNT', 'KMB', 'MPT', 'MSTIL', 'MTUR', 'REPL', 'STB','SBT', 'TEL', 'TTK', 'TNB', 'UNI', 'VITA']

    output_file = 'all_issuers_analysis.txt'

    analyze_all_issuers(issuers, output_file)

if __name__ == '__main__':
    main()
