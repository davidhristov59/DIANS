import pdfplumber
import csv
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from translate import Translator

def extract_text_from_pdf(pdf_file_path):
    text = ""
    try:
        with pdfplumber.open(pdf_file_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text()
    except Exception as e:
        print(f"Error reading {pdf_file_path}: {e}")
    return text

def translate_text(text, src_language='mk', dest_language='en'):
    if not text.strip():
        print("Translation skipped: No text provided.")
        return ""
    translator = Translator(from_lang=src_language, to_lang=dest_language)
    try:
        translated_text = translator.translate(text)
        return translated_text
    except Exception as e:
        print(f"Error translating text: {e}")
        return ""


def perform_sentiment_analysis(text):
    analyzer = SentimentIntensityAnalyzer()
    sentiment_scores = analyzer.polarity_scores(text)
    return sentiment_scores

def make_stock_recommendation(sentiment_scores):
    compound_score = sentiment_scores['compound']

    if compound_score > 0.1:
        recommendation = "Купи"  # Buy
        predicted_price_change = "ќе се зголеми"  # will increase
    elif compound_score < -0.1:
        recommendation = "Продади"  # Sell
        predicted_price_change = "ќе се намали"  # will decrease
    else:
        recommendation = "Задржи"  # Hold
        predicted_price_change = "ќе остане стабилна"  # will remain stable

    return recommendation, predicted_price_change

def process_issuer(issuer_name):
    pdf_file_paths = [
        f'sentimentalAnalysis/{issuer_name}/{issuer_name}.pdf',
        f'sentimentalAnalysis/{issuer_name}/{issuer_name}2023.pdf',
        f'sentimentalAnalysis/{issuer_name}/{issuer_name}2022.pdf'
    ]

    combined_text = ""
    for pdf_file_path in pdf_file_paths:
        print(f"Extracting text from {pdf_file_path}...")
        pdf_text = extract_text_from_pdf(pdf_file_path)
        combined_text += pdf_text + " "

    if combined_text.strip():
        print(f"Translating text for sentiment analysis...")
        translated_text = translate_text(combined_text)

        print(f"Performing sentiment analysis for {issuer_name}...")
        sentiment_scores = perform_sentiment_analysis(translated_text)

        recommendation, predicted_price_change = make_stock_recommendation(sentiment_scores)

        description = (
            f"Сентиментални резултати: {sentiment_scores}\n"
            f"Предвидување: Цената на акциите {predicted_price_change}.\n"
            f"Препорака: {recommendation} акции."
        )

        result = {
            "issuer": issuer_name,
            "description": description
        }

        return result
    else:
        return {
            "issuer": issuer_name,
            "description": "Грешка при обработката или нема извлечен текст."  # Error in processing or no text extracted.
        }

def analyze_all_issuers(issuers, output_file):
    print(f"Започнување на анализа за {len(issuers)} издавачи...")

    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.DictWriter(csvfile, fieldnames=["Име на издавач", "Опис"])
        csv_writer.writeheader()

        for issuer in issuers:
            result = process_issuer(issuer)
            csv_writer.writerow({"Име на издавач": result['issuer'], "Опис": result['description']})

    print(f"Резултатите се зачувани во {output_file}")

def main():
    issuers = ['ALK', 'CKB', 'GRNT', 'KMB', 'MPT', 'MSTIL', 'MTUR', 'REPL',
               'STB', 'SBT', 'TEL', 'TTK', 'TNB', 'UNI', 'VITA', 'OKTA']

    output_file = 'analysis_results.csv'

    analyze_all_issuers(issuers, output_file)

if __name__ == '__main__':
    main()
