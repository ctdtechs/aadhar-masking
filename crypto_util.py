from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
import base64
import json

AES_KEY = base64.b64decode('uodoJA3L10mZstPZmvB0fp0McXkE7LWiWlxmEVp/QUk=')
AES_IV = base64.b64decode('exHWhMYkxb3t4rxcLUzX9g==')

def encrypt_payload(data: dict) -> str:
    """
    Encrypt a dictionary payload and return a Base64-encoded string.
    """
    json_data = json.dumps(data).encode('utf-8')
    cipher = AES.new(AES_KEY, AES.MODE_CBC, AES_IV)
    ct_bytes = cipher.encrypt(pad(json_data, AES.block_size))
    return base64.b64encode(ct_bytes).decode('utf-8')

def decrypt_payload(encrypted_data: str) -> dict:
    """
    Decrypt a Base64-encoded encrypted string back to a dictionary.
    """
    ct_bytes = base64.b64decode(encrypted_data)
    cipher = AES.new(AES_KEY, AES.MODE_CBC, AES_IV)
    pt = unpad(cipher.decrypt(ct_bytes), AES.block_size)
    return json.loads(pt.decode('utf-8'))