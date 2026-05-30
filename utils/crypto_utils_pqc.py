"""
Post-Quantum Cryptography Utilities

Provides secure key management using NIST-approved post-quantum cryptography:

1. ML-KEM (Kyber768) - Key Encapsulation Mechanism
   - Protects master_seed from quantum attacks
   - secure_key_export() - Wrap master_seed
   - secure_key_import() - Unwrap master_seed

2. ML-DSA (Dilithium3) - Digital Signature Scheme
   - Ensures metadata integrity and authenticity
   - sign_bundle() - Sign metadata with sender's secret key
   - verify_bundle() - Verify signature with sender's public key
"""

import os
import base64
import sys
import io
from typing import Tuple, Dict

# Suppress subprocess output from oqs installation attempts
old_stderr = sys.stderr
sys.stderr = io.StringIO()

try:
    try:
        import oqs
        OQS_AVAILABLE = True
    except BaseException as e:  # Use BaseException to catch SystemExit as well
        OQS_AVAILABLE = False
        error_msg = str(e)
        if "No oqs shared libraries found" in error_msg or "RuntimeError" in str(type(e)):
            pass  # Silently fall back
        else:
            pass  # Silently fall back
finally:
    sys.stderr = old_stderr

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes

from utils.logger import setup_logger, get_config_path
from utils.bench import measure

logger = setup_logger("CRYPTO_PQC", get_config_path())


def generate_kyber_keypair() -> Tuple[bytes, bytes]:
    """
    Generate ML-KEM (Kyber768) keypair for post-quantum key encapsulation.
    
    Returns:
        Tuple of (public_key, private_key) in bytes
    
    Raises:
        RuntimeError: If liboqs-python not available
    """
    if not OQS_AVAILABLE:
        raise RuntimeError("liboqs-python not installed. Cannot generate Kyber keys.")
    
    logger.info("🔐 Generating ML-KEM (Kyber768) keypair...")
    
    with oqs.KeyEncapsulation("ML-KEM-768") as kem:
        with measure("ML-KEM key generation"):
            public_key = kem.generate_keypair()
        # kem.export_secret_key() returns the private key
        # Note: liboqs-python handles key generation internally
    
    logger.info("✅ ML-KEM keypair generated (Kyber768)")
    return public_key


def secure_key_export(master_seed: bytes, recipient_public_key: bytes) -> Dict[str, str]:
    """
    Wrap master_seed with ML-KEM before transmission/storage.
    
    Performs KEM encapsulation using recipient's public key, derives a wrapping key
    from the shared secret, and AES-256-GCM encrypts the master_seed.
    
    Args:
        master_seed: 32-byte encryption seed to protect
        recipient_public_key: Recipient's ML-KEM (Kyber768) public key (bytes)
    
    Returns:
        dict with:
            - "kem_ciphertext": hex string (send to recipient)
            - "wrapped_seed": hex string (send to recipient)
            - "wrap_nonce": hex string (send to recipient)
            - "kem_algorithm": "Kyber768"
    
    Raises:
        RuntimeError: If liboqs-python not available or encapsulation fails
    """
    if not OQS_AVAILABLE:
        raise RuntimeError("liboqs-python not installed. Cannot perform ML-KEM wrapping.")
    
    if len(master_seed) != 32:
        raise ValueError(f"master_seed must be 32 bytes, got {len(master_seed)}")
    
    logger.info("🔐 Performing ML-KEM key encapsulation (Kyber768)...")
    
    try:
        with oqs.KeyEncapsulation("ML-KEM-768") as kem:
            with measure("ML-KEM encapsulation"):
                kem_ciphertext, shared_secret = kem.encap_secret(recipient_public_key)
    except Exception as e:
        logger.error(f"❌ ML-KEM encapsulation failed: {e}")
        raise RuntimeError(f"ML-KEM encapsulation failed: {e}")
    
    # Step 2: Derive wrapping key from shared secret using HKDF-SHA256
    wrapping_key = HKDF(
        algorithm=hashes.SHA256(),
        length=32,  # 256-bit key for AES-GCM
        salt=None,
        info=b"master_seed_wrap"
    ).derive(shared_secret)
    
    # Step 3: Wrap master_seed with AES-256-GCM
    nonce = os.urandom(12)
    aesgcm = AESGCM(wrapping_key)
    wrapped_seed = aesgcm.encrypt(nonce, master_seed, None)
    
    logger.info(
        f"✅ Master seed wrapped with ML-KEM: "
        f"KEM={len(kem_ciphertext)} bytes, Wrapped={len(wrapped_seed)} bytes, Nonce={len(nonce)} bytes"
    )
    
    return {
        "kem_ciphertext": kem_ciphertext.hex(),
        "wrapped_seed": wrapped_seed.hex(),
        "wrap_nonce": nonce.hex(),
        "kem_algorithm": "Kyber768"
    }


def secure_key_import(kem_ciphertext: str, wrapped_seed: str, wrap_nonce: str, 
                     recipient_private_key: bytes) -> bytes:
    """
    Recover master_seed using recipient's ML-KEM private key.
    
    Performs KEM decapsulation to recover the shared secret, derives the wrapping key,
    and AES-256-GCM decrypts the master_seed.
    
    Args:
        kem_ciphertext: hex string from sender (KEM ciphertext)
        wrapped_seed: hex string from sender (wrapped master_seed)
        wrap_nonce: hex string from sender (AES-GCM nonce)
        recipient_private_key: Recipient's ML-KEM (Kyber768) private key (bytes)
    
    Returns:
        master_seed: 32-byte decrypted encryption seed
    
    Raises:
        RuntimeError: If liboqs-python not available or decapsulation fails
        ValueError: If KEM data is malformed
    """
    if not OQS_AVAILABLE:
        raise RuntimeError("liboqs-python not installed. Cannot perform ML-KEM unwrapping.")
    
    logger.info("🔓 Performing ML-KEM key decapsulation (Kyber768)...")
    
    try:
        # Step 1: Decapsulate with private key to recover shared secret
        # liboqs-python 0.14+: secret_key is passed to constructor, not decap_secret
        with oqs.KeyEncapsulation("ML-KEM-768", secret_key=recipient_private_key) as kem:
            with measure("ML-KEM decapsulation"):
                shared_secret = kem.decap_secret(bytes.fromhex(kem_ciphertext))
    except Exception as e:
        logger.error(f"❌ ML-KEM decapsulation failed: {e}")
        raise RuntimeError(f"ML-KEM decapsulation failed: {e}")
    
    # Step 2: Derive wrapping key from shared secret (same HKDF parameters)
    wrapping_key = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=None,
        info=b"master_seed_wrap"
    ).derive(shared_secret)
    
    # Step 3: Unwrap master_seed with AES-256-GCM
    try:
        aesgcm = AESGCM(wrapping_key)
        master_seed = aesgcm.decrypt(
            bytes.fromhex(wrap_nonce),
            bytes.fromhex(wrapped_seed),
            None
        )
    except Exception as e:
        logger.error(f"❌ Master seed decryption failed (authentication tag mismatch?): {e}")
        raise RuntimeError(f"Master seed decryption failed: {e}")
    
    if len(master_seed) != 32:
        raise ValueError(f"Recovered master_seed has unexpected length: {len(master_seed)} (expected 32)")
    
    logger.info(f"✅ Master seed recovered from ML-KEM: {len(master_seed)} bytes")
    return master_seed


def save_pqc_keys_to_file(wrapped_keys: dict, output_path: str) -> None:
    """
    Save ML-KEM wrapped keys to JSON file.
    
    Args:
        wrapped_keys: dict from secure_key_export()
        output_path: Path to save keys.json
    """
    import json
    
    pqc_metadata = {
        "key_encapsulation": {
            "algorithm": "Kyber768 (ML-KEM NIST-approved)",
            "kem_ciphertext": wrapped_keys["kem_ciphertext"],
            "wrapped_seed_nonce": wrapped_keys["wrap_nonce"],
            "wrapped_seed": wrapped_keys["wrapped_seed"],
            "version": "2.0",
            "created_at": __import__("datetime").datetime.now().isoformat()
        },
        "important": "Master seed is wrapped with ML-KEM. Never in plaintext. Use recipient's ML-KEM private key with secure_key_import() to recover."
    }
    
    with open(output_path, "w") as f:
        json.dump(pqc_metadata, f, indent=2)
    
    logger.info(f"✅ PQC-protected keys saved: {output_path}")


def load_pqc_keys_from_file(keys_path: str) -> dict:
    """
    Load ML-KEM wrapped keys from JSON file.
    
    Args:
        keys_path: Path to keys.json file
    
    Returns:
        dict with kem_ciphertext, wrapped_seed, wrap_nonce
    """
    import json
    
    with open(keys_path, "r") as f:
        data = json.load(f)
    
    pqc = data.get("key_encapsulation", {})
    
    return {
        "kem_ciphertext": pqc.get("kem_ciphertext"),
        "wrapped_seed": pqc.get("wrapped_seed"),
        "wrap_nonce": pqc.get("wrapped_seed_nonce")
    }


# ═════════════════════════════════════════════════════════════════════════════
# ML-DSA (Dilithium3) - Digital Signatures for Metadata Integrity
# ═════════════════════════════════════════════════════════════════════════════


def generate_dilithium_keypair() -> Tuple[bytes, bytes]:
    """
    Generate ML-DSA (Dilithium3) keypair for metadata signing.
    
    Returns:
        Tuple of (public_key, private_key) in bytes
    
    Raises:
        RuntimeError: If liboqs-python not available
    """
    if not OQS_AVAILABLE:
        raise RuntimeError("liboqs-python not installed. Cannot generate Dilithium keys.")
    
    logger.info("🔐 Generating ML-DSA (Dilithium3) keypair...")
    
    try:
        with oqs.Signature("ML-DSA-65") as sig:
            with measure("ML-DSA key generation"):
                public_key = sig.generate_keypair()
                private_key = sig.export_secret_key()
    except Exception as e:
        logger.error(f"❌ ML-DSA keypair generation failed: {e}")
        raise RuntimeError(f"ML-DSA keypair generation failed: {e}")
    
    logger.info(f"✅ ML-DSA keypair generated (Dilithium3): public={len(public_key)}B, private={len(private_key)}B")
    return public_key, private_key


def sign_bundle(metadata_path: str, sender_private_key: bytes) -> str:
    """
    Sign metadata bundle with ML-DSA (Dilithium3).
    
    Ensures metadata integrity and sender authenticity. Signature should be
    verified by recipient before any decryption attempt.
    
    Args:
        metadata_path: Path to metadata.json file
        sender_private_key: Sender's ML-DSA private key (bytes)
    
    Returns:
        Signature as hex string (should be saved to .sig file)
    
    Raises:
        RuntimeError: If signing fails
        FileNotFoundError: If metadata file not found
    """
    if not OQS_AVAILABLE:
        raise RuntimeError("liboqs-python not installed. Cannot sign bundle.")
    
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    
    logger.info(f"🔓 Signing metadata bundle: {metadata_path}")
    
    try:
        # Read metadata file
        with open(metadata_path, "rb") as f:
            content = f.read()
        
        # Sign with Dilithium3
        # liboqs-python 0.14+: secret_key is passed to constructor, not sign()
        with oqs.Signature("ML-DSA-65", secret_key=sender_private_key) as signer:
            with measure("ML-DSA signing"):
                signature = signer.sign(content)
        
        signature_hex = signature.hex()
        logger.info(f"✅ Metadata signed with ML-DSA (Dilithium3): signature={len(signature)} bytes")
        return signature_hex
    
    except Exception as e:
        logger.error(f"❌ Metadata signing failed: {e}")
        raise RuntimeError(f"Metadata signing failed: {e}")


def verify_bundle(metadata_path: str, signature_hex: str, sender_public_key: bytes) -> bool:
    """
    Verify metadata bundle signature with ML-DSA (Dilithium3).
    
    Authenticates metadata and ensures it was not tampered with. Should be
    called before any decryption attempt.
    
    Args:
        metadata_path: Path to metadata.json file
        signature_hex: Signature as hex string (from .sig file)
        sender_public_key: Sender's ML-DSA public key (bytes)
    
    Returns:
        True if signature is valid, False otherwise
    
    Raises:
        RuntimeError: If verification process fails
        FileNotFoundError: If metadata file not found
    """
    if not OQS_AVAILABLE:
        raise RuntimeError("liboqs-python not installed. Cannot verify bundle.")
    
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    
    logger.info(f"🔍 Verifying metadata bundle signature: {metadata_path}")
    
    try:
        # Read metadata file
        with open(metadata_path, "rb") as f:
            content = f.read()
        
        # Verify with Dilithium3
        with oqs.Signature("ML-DSA-65") as verifier:
            with measure("ML-DSA verification"):
                is_valid = verifier.verify(
                    content,
                    bytes.fromhex(signature_hex),
                    sender_public_key
                )
        
        if is_valid:
            logger.info("✅ Metadata signature verified (ML-DSA Dilithium3)")
        else:
            logger.warning("❌ Metadata signature verification FAILED - Bundle may be tampered")
        
        return is_valid
    
    except Exception as e:
        logger.error(f"❌ Metadata signature verification failed: {e}")
        return False


def save_signature_file(signature_hex: str, output_path: str) -> None:
    """
    Save signature to file (.sig extension).
    
    Args:
        signature_hex: Signature as hex string
        output_path: Path to save signature file
    """
    with open(output_path, "w") as f:
        f.write(signature_hex)
    logger.info(f"✅ Signature saved: {output_path}")


def load_signature_file(sig_path: str) -> str:
    """
    Load signature from file (.sig extension).
    
    Args:
        sig_path: Path to signature file
    
    Returns:
        Signature as hex string
    
    Raises:
        FileNotFoundError: If signature file not found
    """
    if not os.path.exists(sig_path):
        raise FileNotFoundError(f"Signature file not found: {sig_path}")
    
    with open(sig_path, "r") as f:
        signature_hex = f.read().strip()
    
    logger.info(f"✅ Signature loaded: {sig_path}")
    return signature_hex


def save_dilithium_keys(public_key: bytes, private_key: bytes, public_key_path: str, private_key_path: str) -> None:
    """
    Save Dilithium keypair to files.
    
    Args:
        public_key: Public key bytes
        private_key: Private key bytes
        public_key_path: Path to save public key
        private_key_path: Path to save private key
    """
    # Save public key
    os.makedirs(os.path.dirname(public_key_path), exist_ok=True)
    with open(public_key_path, "wb") as f:
        f.write(public_key)
    
    # Save private key
    os.makedirs(os.path.dirname(private_key_path), exist_ok=True)
    with open(private_key_path, "wb") as f:
        f.write(private_key)
    
    logger.info(f"✅ Dilithium keypair saved: public={public_key_path}, private={private_key_path}")


def load_dilithium_public_key(public_key_path: str) -> bytes:
    """
    Load Dilithium public key from file.
    
    Args:
        public_key_path: Path to public key file
    
    Returns:
        Public key bytes
    
    Raises:
        FileNotFoundError: If file not found
    """
    if not os.path.exists(public_key_path):
        raise FileNotFoundError(f"Public key file not found: {public_key_path}")
    
    with open(public_key_path, "rb") as f:
        public_key = f.read()
    
    return public_key


def load_dilithium_private_key(private_key_path: str) -> bytes:
    """
    Load Dilithium private key from file.
    
    Args:
        private_key_path: Path to private key file
    
    Returns:
        Private key bytes
    
    Raises:
        FileNotFoundError: If file not found
    """
    if not os.path.exists(private_key_path):
        raise FileNotFoundError(f"Private key file not found: {private_key_path}")
    
    with open(private_key_path, "rb") as f:
        private_key = f.read()
    
    return private_key


# ═════════════════════════════════════════════════════════════════════════════
# Key Protection at Rest - Password-based Encryption (Scrypt + AES-256-GCM)
# ═════════════════════════════════════════════════════════════════════════════


def protect_keys(keys: dict, passphrase: str) -> bytes:
    """
    Encrypt cryptographic keys at rest using password-based key derivation.
    
    Uses Scrypt for key derivation and AES-256-GCM for encryption.
    Never store raw key material - this function protects keys in storage.
    
    Args:
        keys: Dictionary containing key material (master_seed, aes_key, salt, etc.)
        passphrase: User-provided passphrase for key derivation
    
    Returns:
        Encrypted key blob (salt + nonce + ciphertext) as bytes
        Safe to store in files, transmit over network
    
    Raises:
        ValueError: If passphrase is empty or keys dict is empty
    """
    if not passphrase:
        raise ValueError("Passphrase cannot be empty")
    if not keys:
        raise ValueError("Keys dictionary cannot be empty")
    
    logger.info("🔐 Encrypting key material at rest (Scrypt + AES-256-GCM)...")
    
    from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
    import json
    
    try:
        # Step 1: Generate random salt for Scrypt
        salt = os.urandom(16)
        
        # Step 2: Derive Key Encryption Key (KEK) using Scrypt
        # Parameters: n=2^14 (memory cost), r=8, p=1 (CPU/memory balance)
        kdf = Scrypt(
            salt=salt,
            length=32,  # 256-bit KEK for AES
            n=2**14,    # Memory cost (16,384 iterations)
            r=8,        # Block size
            p=1         # Parallelization
        )
        with measure("Scrypt derivation"):
            kek = kdf.derive(passphrase.encode('utf-8'))
        
        # Step 3: Serialize keys to JSON
        plaintext = json.dumps(keys).encode('utf-8')
        
        # Step 4: Generate random nonce for AES-GCM
        nonce = os.urandom(12)
        
        # Step 5: Encrypt with AES-256-GCM using KEK
        aesgcm = AESGCM(kek)
        ciphertext = aesgcm.encrypt(nonce, plaintext, None)
        
        # Step 6: Combine salt + nonce + ciphertext into single blob
        encrypted_blob = salt + nonce + ciphertext
        
        logger.info(
            f"✅ Keys encrypted at rest: "
            f"salt={len(salt)}B, nonce={len(nonce)}B, ciphertext={len(ciphertext)}B, "
            f"total={len(encrypted_blob)}B"
        )
        
        return encrypted_blob
    
    except Exception as e:
        logger.error(f"❌ Key encryption failed: {e}")
        raise RuntimeError(f"Key protection failed: {e}")


def unprotect_keys(encrypted_blob: bytes, passphrase: str) -> dict:
    """
    Decrypt cryptographic keys protected at rest using password.
    
    Reverses the protect_keys() operation: derives KEK from passphrase,
    then AES-GCM decrypts the blob to recover original keys.
    
    Args:
        encrypted_blob: Encrypted key blob (salt + nonce + ciphertext)
        passphrase: User-provided passphrase for key derivation
    
    Returns:
        Decrypted keys dictionary (master_seed, aes_key, salt, etc.)
    
    Raises:
        ValueError: If blob format invalid, passphrase wrong, or decryption fails
    """
    if not passphrase:
        raise ValueError("Passphrase cannot be empty")
    if len(encrypted_blob) < 28:  # Minimum: 16 bytes salt + 12 bytes nonce
        raise ValueError(f"Encrypted blob too short: {len(encrypted_blob)} bytes (minimum 28)")
    
    logger.info("🔓 Decrypting key material at rest (Scrypt + AES-256-GCM)...")
    
    from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
    import json
    
    try:
        # Step 1: Extract salt, nonce, ciphertext from blob
        salt = encrypted_blob[:16]           # First 16 bytes
        nonce = encrypted_blob[16:28]        # Next 12 bytes
        ciphertext = encrypted_blob[28:]     # Remaining bytes
        
        # Step 2: Derive KEK using same Scrypt parameters
        kdf = Scrypt(
            salt=salt,
            length=32,
            n=2**14,
            r=8,
            p=1
        )
        with measure("Scrypt derivation"):
            kek = kdf.derive(passphrase.encode('utf-8'))
        
        # Step 3: Decrypt with AES-256-GCM using KEK
        aesgcm = AESGCM(kek)
        plaintext = aesgcm.decrypt(nonce, ciphertext, None)
        
        # Step 4: Deserialize JSON to dict
        keys = json.loads(plaintext.decode('utf-8'))
        
        logger.info(f"✅ Keys decrypted from rest: recovered {len(keys)} key items")
        return keys
    
    except Exception as e:
        logger.error(f"❌ Key decryption failed: {e}")
        raise RuntimeError(f"Key unprotection failed (wrong passphrase?): {e}")


def save_protected_keys(keys: dict, passphrase: str, output_path: str) -> None:
    """
    Save encrypted keys to file with protection.
    
    Args:
        keys: Dictionary containing key material
        passphrase: User passphrase for encryption
        output_path: Path to save encrypted key blob
    """
    encrypted_blob = protect_keys(keys, passphrase)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(encrypted_blob)
    
    logger.info(f"✅ Protected keys saved to: {output_path}")


def load_protected_keys(input_path: str, passphrase: str) -> dict:
    """
    Load and decrypt keys from protected file.
    
    Args:
        input_path: Path to encrypted key blob file
        passphrase: User passphrase for decryption
    
    Returns:
        Decrypted keys dictionary
    
    Raises:
        FileNotFoundError: If file not found
        RuntimeError: If decryption fails
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Protected keys file not found: {input_path}")
    
    with open(input_path, "rb") as f:
        encrypted_blob = f.read()
    
    keys = unprotect_keys(encrypted_blob, passphrase)
    logger.info(f"✅ Protected keys loaded from: {input_path}")
    return keys
