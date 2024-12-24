import pdfplumber
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# Function to extract text from the PDF file
def extract_text_from_pdf(pdf_file_path):
    text = ""
    with pdfplumber.open(pdf_file_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text


# Function to perform sentiment analysis using VADER
def perform_sentiment_analysis(text):
    analyzer = SentimentIntensityAnalyzer()
    sentiment_scores = analyzer.polarity_scores(text)
    return sentiment_scores


# Function to make stock recommendation based on sentiment
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


# Main function
def main():
    # Path to the ALK.pdf file
    pdf_file_path = 'sentimentalAnalysis/CKB.pdf'

    # Step 1: Extract text from the PDF
    print("Extracting text from PDF...")
    pdf_text = extract_text_from_pdf(pdf_file_path)

    # Step 2: Perform sentiment analysis
    print("Performing sentiment analysis...")
    sentiment_scores = perform_sentiment_analysis(pdf_text)
    print(f"Sentiment Scores: {sentiment_scores}")

    # Step 3: Make stock recommendation
    recommendation, predicted_price_change = make_stock_recommendation(sentiment_scores)

    # Display results
    print(f"Predicted stock price will {predicted_price_change}.")
    print(f"Recommendation: {recommendation} stocks.")


if __name__ == '__main__':
    main()
