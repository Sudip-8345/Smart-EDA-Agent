import io

def df_info_string(df, max_row=5):
    buf = io.StringIO()
    df.info(buf)
    schema = buf.getvalue()
    head = df.head(max_row).to_markdown(index=False)

    missing = df.isna().sum()
    missing = missing[missing > 0]
    if missing.empty:
        missing_info = "No missing Values."
    else:
        missing_info = str(missing)
    return f"###schema:\n'''\n{schema}'''\n\n###Preview:\n{head}\n\n###Missing:\n{missing_info}"