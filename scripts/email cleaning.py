def extract_email_addresses(text):
    pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'
    emails = re.findall(pattern, text)
    return emails if emails else None

def remove_emails(text, emails):
    for email in emails.dropna():
        for string in email:
            text = re.sub(re.escape(string), '', text)
    return text

def normalize_document(doc):
    wpt = nltk.WordPunctTokenizer()
    stop_words = set(nltk.corpus.stopwords.words('english'))

    # lowercase and remove special characters\whitespace
    doc = re.sub(r'[^a-zA-Z\s]', '', str(doc), re.I | re.A)
    doc = doc.lower()
    doc = doc.strip()
    
    # extract email addresses using a more robust regex
    emails = extract_email_addresses(doc)

    # remove email addresses from the document
    doc = remove_emails(doc, emails)
            
    # tokenize document
    tokens = wpt.tokenize(doc)
    # filter stopwords out of document
    filtered_tokens = [token for token in tokens if token not in stop_words]
    # re-create document from filtered tokens
    doc = ' '.join(filtered_tokens)
    
    return doc

# Function to apply cleaning to an entire DataFrame
def clean_dataframe(df):
    # Apply the cleaning function to each cell in the DataFrame
    cleaned_df = df.applymap(lambda x: normalize_document(x))
    return cleaned_df
