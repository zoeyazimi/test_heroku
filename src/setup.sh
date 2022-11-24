mkdir -p ~/.streamlit/

echo "\
[general]\n\
email = \"z.azimi@ixp-duesseldorf.de\"\n\
" > ~/.streamlit/credentials.toml

echo "\
[server]\n\
headless = true\n\
port = $PORT\n\
enableCORS=false\n\
\[theme]\n\
primaryColor = \"#44369a\"\n\
backgroundColor = \"#6c6c71\"\n\
secondaryBackgroundColor = \"#1a1b22\"\n\
textColor = \"#ffffff\"\n\
font = \"sans serif\"\n\
" > ~/.streamlit/config.toml
