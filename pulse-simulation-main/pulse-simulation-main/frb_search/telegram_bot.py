import requests
from tqdm.contrib.telegram import tqdm
from .config import config

# Retrieve the values from the config file
TOKEN = config['DEFAULT']['telegram_token'] if config.has_option('DEFAULT', 'telegram_token') else None
CHAT_ID = int(config['DEFAULT']['telegram_id']) if config.has_option('DEFAULT', 'telegram_id') and config['DEFAULT']['telegram_id'] else None

DOCTYPES = ['Video', 'Photo', 'Document']

def url(token: str = None, method: str = None) -> str:
    """Returns the url for the given method
    """
    if token is None:
        token = TOKEN

    return f'https://api.telegram.org/bot{token}/{method}'


def get_updates(token: str = None, offset: int = None, timeout: int = 60) -> dict:
    """Get updates from Telegram API
    """
    if token is None:
        token = TOKEN

    params = {'offset': offset, 'timeout': timeout}
    return requests.get(url(token, 'getUpdates'), params=params).json()


def send_message(text: str, token: str = None, chat_id: int = None, parse_mode: str = 'MarkdownV2', timeout=3) -> dict:
    """Send message to Telegram API
    """
    if token is None:
        token = TOKEN

    if chat_id is None:
        chat_id = CHAT_ID

    if not token or not chat_id:
        return

    params = {'chat_id': chat_id, 'text': text, 'parse_mode': parse_mode}
    try:
        response = requests.post(url(token, 'sendMessage'), params=params, timeout=timeout)
        response.raise_for_status()  # Raise an exception for HTTP errors
        return response.json()
    except requests.Timeout:
        print("Connection timed out. Check your internet connection.")
        return
    except requests.RequestException as e:
        print(f"Error sending message: {e}")
        return


def send(attachment: str, token: str = None, chat_id: int = None,
         doc_type: str = 'Document', caption: str = None, **kwargs) -> dict:
    """Send attachment to Telegram API.
    """
    if token is None:
        token = TOKEN

    if chat_id is None:
        chat_id = CHAT_ID

    if doc_type not in DOCTYPES:
        raise ValueError(f'{doc_type} is invalid. Must be one of {DOCTYPES}')

    params = {'chat_id': chat_id, 'caption': caption} | kwargs
    with open(attachment, 'rb') as f:
        files = {doc_type.lower(): f.read()}

    return requests.post(url(token, f'send{doc_type}'), params=params, files=files).json()


def send_progress(iterable, token: str = None, chat_id : int = None, **kwargs) -> tqdm:
    """Send progress bar to Telegram API. Returns a tqdm object.

    Args:
        iterable: Iterable to wrap around.
        token: Telegram token.
        chat_id: Telegram chat id.
        **kwargs: Keyword arguments to pass to tqdm.contrib.telegram.tqdm.
    """
    if token is None:
        token = TOKEN

    if chat_id is None:
        chat_id = CHAT_ID

    return tqdm(iterable, token=token, chat_id=chat_id, **kwargs)


if __name__ == '__main__':
    
    send_message('Hello world\!')
    print('xd')
    