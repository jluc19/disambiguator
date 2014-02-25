def format_for_csv(string):
    """
    Replaces special characters used by comma separated value (CSV) files
    with their HTML equivalents.
    """
    string = string.strip()
    string = string.replace('\n', ' ')
    string = string.replace('&', "&amp;")
    string = string.replace(',', "&#44;")
    string = string.replace('>', "&gt;")
    string = string.replace('<', "&lt;")
    string = string.replace('"', "&quot;")
    string = string.replace("'", "&#39;")
    return string