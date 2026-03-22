import os
import hashlib
import requests
import tarfile
import zipfile
import gzip
import shutil
import subprocess
import boto3
import configparser
import gdown

from abc import ABC, abstractmethod
from botocore import UNSIGNED
from botocore.config import Config
from botocore.exceptions import NoCredentialsError, ClientError
from google.cloud import storage
from sklearn.datasets import get_data_home
from typing import Optional, Tuple, Union


class BaseDataset(ABC):
    """
    Abstract dataset class for fetching and extracting data.

    Supports fetching from HTTP, wget, S3 (authenticated and anonymous,
    including OpenNeuro), Google Drive, and now Google Cloud Storage.  Extracts various archive formats
    (tar, zip, gz, etc.).  Subclasses should implement __len__ and __getitem__
    for data access. Handles AWS credentials from a configuration file.
    """

    def __init__(self, root_dir: Optional[str] = None):
        """
        Initialize the BaseDataset.

        Args:
            root_dir: Root directory for downloads/extraction.  If None,
                      uses scikit-learn's default data home.
        """
        if root_dir is None:
            self.root_dir = self.home()  # Use default data home
        else:
            self.root_dir = root_dir
        self.aws_config = self._load_aws_config()  # Load AWS config

    def _load_aws_config(self) -> Optional[configparser.ConfigParser]:
        """Load AWS configuration from a .ini file."""
        config_path = os.environ.get('AWS_CONFIG_FILE')  # Get path from env
        if not config_path:
            print(
                "Warning: AWS_CONFIG_FILE environment variable not set. "
                "Authenticated S3 downloads may fail."
            )
            return None

        if not os.path.exists(config_path):
            print(
                f"Warning: AWS config file not found at {config_path}. "
                "Authenticated S3 downloads may fail."
            )
            return None

        config = configparser.ConfigParser()
        try:
            config.read(config_path)
            if 'default' not in config:  # Check for default section
                print(
                    "Warning: [default] section missing in AWS config. "
                    "Authenticated S3 downloads may fail."
                )
                return None

            # Check for required keys *only* if using credentials
            required_keys = [
                'aws_access_key_id', 'aws_secret_access_key', 'region_name'
            ]
            if any(key in config['default'] for key in required_keys):
                if not all(key in config['default'] for key in required_keys):
                    print(
                        "Warning: Missing AWS credentials in config file. "
                        "Authenticated S3 downloads may fail."
                    )
                    return None
            return config

        except configparser.Error as e:
            print(f"Error parsing AWS config file: {e}")
            return None

    def home(self, *suffix_paths: str) -> str:
        """Return the path to the dataset's home directory."""
        home_path = os.environ.get('SCIKIT_LEARN_DATA')  # Get path from env
        if not home_path:
            raise ValueError(
                f"SCIKIT_LEARN_DATA environment variable must be setup for fetching data (export SCIKIT_LEARN_DATA=path/to/download/data)"
            )
        return os.path.join(
            "/content/drive/MyDrive/Psych249", self.__class__.__name__, *suffix_paths
        )

    @abstractmethod
    def __len__(self):
        """Return the total number of samples in the dataset."""
        pass

    @abstractmethod
    def __getitem__(self, idx):
        """Return a single sample from the dataset at index idx."""
        pass

    def fetch(
        self,
        source: str,
        target_dir: Optional[str] = None,
        filename: Optional[str] = None,
        method: str = 'auto',
        force_download: bool = False,
        **kwargs,
    ) -> str:
        """
        Fetch data from a source (http, wget, s3, gdown, or google cloud storage).

        Args:
            source: Source URL (http/https/gdrive/gs://) or s3:// path.
            target_dir: Download directory (defaults to self.root_dir).
            filename: Filename to save as (defaults to source's last part).
            method: Download method ('auto', 'http', 'wget', 's3', 'gdown', 'gcs').
            force_download: Force download even if the file exists.
            **kwargs: Additional arguments for download methods.

        Returns:
            The path to the downloaded file.
        """
        if target_dir is None:
            target_dir = self.root_dir
        os.makedirs(target_dir, exist_ok=True)

        # Derive filename if not provided
        if filename is None:
            if '/' in source:
                filename = source.split('/')[-1]
                if not filename:  # Handle URLs ending with '/'
                    filename = hashlib.md5(source.encode()).hexdigest()
            else:
                filename = source

        filepath = os.path.join(target_dir, filename)

        # Skip if file exists and no force_download
        if os.path.exists(filepath) and not force_download:
            print(f"File already exists at {filepath}")
            return filepath

        # Auto-determine download method
        if method == 'auto':
            if source.startswith('s3://'):
                method = 's3'  # S3 handles both authenticated and anonymous
            elif source.startswith('gs://'):
                method = 'gcs'
            elif source.startswith('http://') or source.startswith('https://'):
                if "drive.google.com" in source:  # Google Drive URL
                    method = 'gdown'
                else:
                    try:  # Check for wget
                        subprocess.run(
                            ['wget', '--version'],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            check=True,
                        )
                        method = 'wget'
                    except (subprocess.SubprocessError, FileNotFoundError):
                        method = 'http'
            else:
                raise ValueError(
                    "Cannot auto-determine download method for source:"
                    f" {source}"
                )

        # Dispatch to fetch method
        if method == 'http':
            return self._fetch_http(source, filepath, **kwargs)
        elif method == 'wget':
            return self._fetch_wget(source, filepath, **kwargs)
        elif method == 's3':
            return self._fetch_s3(source, target_dir, filename, **kwargs)
        elif method == 'gdown':
            return self._fetch_gdown(source, filepath, **kwargs)
        elif method == 'gcs':
            return self._fetch_gcs(source, filepath, **kwargs)
        else:
            raise ValueError(f"Unsupported download method: {method}")

    def _fetch_http(
        self, url: str, filepath: str, timeout: int = 300, **kwargs
    ) -> str:
        """Download a file over HTTP/HTTPS using requests."""
        print(f"Downloading {url} to {filepath} using HTTP...")
        response = requests.get(url, stream=True, timeout=timeout)
        response.raise_for_status()

        total_size = response.headers.get('content-length')
        chunk_size = 1024 * 1024  # 1 MB

        if total_size is not None:
            from tqdm import tqdm
            total_size = int(total_size)
            with open(filepath, "wb") as f, tqdm(
                total=total_size, unit='B', unit_scale=True,
                desc=os.path.basename(filepath),
            ) as pbar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    f.write(chunk)
                    pbar.update(len(chunk))
        else:
            with open(filepath, "wb") as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    f.write(chunk)

        return filepath

    def _fetch_wget(
        self, url: str, filepath: str, timeout: int = 300, **kwargs
    ) -> str:
        """Download a file using wget."""
        print(f"Downloading {url} to {filepath} using wget...")
        cmd = ['wget', url, '-O', filepath, '--timeout', str(timeout)]

        # Add wget-specific options from kwargs
        for key, value in kwargs.items():
            if key.startswith('wget_'):
                opt = key[5:].replace('_', '-')
                if isinstance(value, bool):  # Boolean flags
                    if value:
                        cmd.append(f'--{opt}')
                else:
                    cmd.append(f'--{opt}={value}')

        subprocess.run(cmd, check=True)
        return filepath

    def _fetch_s3(
        self,
        s3_path: str,
        target_dir: str,
        filename: Optional[str] = None,
        anonymous: bool = False,
        **kwargs,
    ) -> str:
        """Download a file or directory from S3 using the AWS CLI for
           parallel transfers. Falls back to boto3 if AWS CLI is unavailable.
        """
        print(f"Downloading {s3_path} to {target_dir} using S3...")

        # Parse bucket/key
        path_parts = s3_path.split('//')[1].split('/', 1)
        bucket_name = path_parts[0]
        s3_key = path_parts[1] if len(path_parts) > 1 else ''

        # Determine if this is a directory (ends with /) or single file
        is_directory = s3_key.endswith('/') or not s3_key

        # Try AWS CLI first (parallel transfers by default)
        if self._try_aws_cli_s3(s3_path, target_dir, filename,
                                anonymous, is_directory):
            if is_directory:
                return os.path.join(
                    target_dir, filename or s3_key.rstrip('/').split('/')[-1])
            else:
                return os.path.join(
                    target_dir, filename or s3_key.split('/')[-1])

        # Fall back to boto3
        print("AWS CLI not available, falling back to boto3...")
        return self._fetch_s3_boto3(
            s3_path, target_dir, filename, anonymous, **kwargs)

    def _try_aws_cli_s3(
        self,
        s3_path: str,
        target_dir: str,
        filename: Optional[str] = None,
        anonymous: bool = False,
        is_directory: bool = False,
    ) -> bool:
        """Try downloading via AWS CLI (supports parallel transfers).
           Returns True on success, False if CLI unavailable.
        """
        try:
            subprocess.run(
                ['aws', '--version'],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        except (subprocess.SubprocessError, FileNotFoundError):
            return False

        env = os.environ.copy()
        cmd = ['aws', 's3']

        if anonymous:
            cmd.append('--no-sign-request')

        if is_directory:
            local_dir = os.path.join(
                target_dir,
                filename or s3_path.rstrip('/').split('/')[-1])
            os.makedirs(local_dir, exist_ok=True)
            cmd.extend(['sync', s3_path, local_dir])
        else:
            local_path = os.path.join(
                target_dir, filename or s3_path.split('/')[-1])
            cmd.extend(['cp', s3_path, local_path])

        if not anonymous and self.aws_config:
            env['AWS_ACCESS_KEY_ID'] = self.aws_config['default'].get(
                'aws_access_key_id', '')
            env['AWS_SECRET_ACCESS_KEY'] = self.aws_config['default'].get(
                'aws_secret_access_key', '')
            env['AWS_DEFAULT_REGION'] = self.aws_config['default'].get(
                'region_name', 'us-east-1')

        try:
            subprocess.run(cmd, check=True, env=env)
            return True
        except subprocess.SubprocessError as e:
            print(f"AWS CLI download failed: {e}")
            return False

    def _fetch_s3_boto3(
        self,
        s3_path: str,
        target_dir: str,
        filename: Optional[str] = None,
        anonymous: bool = False,
        **kwargs,
    ) -> str:
        """Fallback: download from S3 using boto3 (sequential)."""
        path_parts = s3_path.split('//')[1].split('/', 1)
        bucket_name = path_parts[0]
        s3_key = path_parts[1] if len(path_parts) > 1 else ''

        # Configure boto3 client
        if anonymous:
            s3_client = boto3.client('s3',
                                     config=Config(signature_version=UNSIGNED))

        elif self.aws_config:
            aws_access_key_id = self.aws_config['default'].get(
                'aws_access_key_id'
            )
            aws_secret_access_key = self.aws_config['default'].get(
                'aws_secret_access_key'
            )
            region_name = self.aws_config['default'].get('region_name')

            s3_client = boto3.client(
                's3',
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                region_name=region_name,
            )
        else:
            raise Exception(
                "AWS configuration not loaded and anonymous=False. "
                "Cannot access S3."
            )

        try:
            # --- File vs. Directory Handling ---
            response = s3_client.list_objects_v2(
                Bucket=bucket_name, Prefix=s3_key, Delimiter='/', MaxKeys=1
            )

            if (  # It's a single file
                'Contents' in response
                and len(response['Contents']) == 1
                and response['Contents'][0]['Key'] == s3_key
            ):
                filepath = os.path.join(
                    target_dir, filename or s3_key.split('/')[-1]
                )
                s3_client.download_file(bucket_name, s3_key, filepath)
                return filepath

            else:  # It's a directory (or prefix)
                prefix = s3_key
                if prefix and not prefix.endswith('/'):
                    prefix += '/'

                paginator = s3_client.get_paginator('list_objects_v2')
                page_iterator = paginator.paginate(
                    Bucket=bucket_name, Prefix=prefix
                )

                # Use filename as directory name, if provided
                downloaded_dir = os.path.join(
                    target_dir, filename or s3_key.split('/')[-1]
                )
                if not os.path.exists(downloaded_dir):
                    os.makedirs(downloaded_dir)

                for page in page_iterator:
                    if 'Contents' in page:
                        for obj in page['Contents']:
                            key = obj['Key']
                            relative_path = os.path.relpath(key, prefix)
                            local_path = os.path.join(
                                downloaded_dir, relative_path
                            )

                            # Ensure intermediate directories exist
                            local_dir = os.path.dirname(local_path)
                            if not os.path.exists(local_dir):
                                os.makedirs(local_dir)
                            s3_client.download_file(
                                bucket_name, key, local_path)
                return downloaded_dir

        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                raise FileNotFoundError(
                    f"The specified S3 object does not exist: {s3_path}"
                )
            else:
                raise

        except NoCredentialsError:
            raise Exception("AWS credentials not available.")
        except Exception as e:
            raise Exception(f"Error during S3 download: {e}")

    def _fetch_gdown(self, url: str, filepath: str, **kwargs) -> str:
        """Download a file from Google Drive using gdown."""
        print(f"Downloading {url} to {filepath} using gdown...")
        try:
            gdown.download(url, filepath, quiet=False, **kwargs)
            return filepath
        except Exception as e:
            raise Exception(f"Error downloading from Google Drive: {e}")

    def _fetch_gcs(self, source: str, filepath: str, **kwargs) -> str:
        """
        Download a file from Google Cloud Storage (GCS).

        Args:
            source: GCS URL in the format gs://bucket_name/path/to/file
            filepath: Local path to save the downloaded file.
            **kwargs: Additional arguments (not used here).

        Returns:
            The path to the downloaded file.
        """
        print(
            f"Downloading {source} to {filepath} using Google Cloud Storage...")
        # Parse the source URL
        parts = source[5:].split('/', 1)  # Remove 'gs://'
        if len(parts) != 2:
            raise ValueError(
                "Invalid GCS URL. Expected format: gs://bucket_name/path/to/file")
        bucket_name, blob_name = parts

        # Initialize the GCS client (assumes credentials are set up via GOOGLE_APPLICATION_CREDENTIALS)
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.download_to_filename(filepath)
        return filepath

    def extract(
        self,
        filepath: str,
        extract_dir: Optional[str] = None,
        format: Optional[str] = None,
        delete_archive: bool = False,
        **kwargs,
    ) -> str:
        """
        Extract an archive (tar, zip, gz, etc.).

        Args:
            filepath: Path to the archive file.
            extract_dir: Directory to extract to (defaults to filepath's dir).
            format: Archive format (inferred from extension if None).
            delete_archive: Delete the archive file after extraction.
            **kwargs: Format-specific options.

        Returns:
            The path to the extracted directory or uncompressed file.
        """
        if extract_dir is None:
            extract_dir = os.path.dirname(filepath)
        os.makedirs(extract_dir, exist_ok=True)

        # Auto-detect format if not provided
        if format is None:
            if filepath.endswith('.tar.gz') or filepath.endswith('.tgz'):
                format = 'tar.gz'
            elif filepath.endswith('.tar.bz2') or filepath.endswith('.tbz2'):
                format = 'tar.bz2'
            elif filepath.endswith('.tar'):
                format = 'tar'
            elif filepath.endswith('.zip'):
                format = 'zip'
            elif filepath.endswith('.gz') and not filepath.endswith('.tar.gz'):
                format = 'gz'
            elif filepath.endswith('.rar'):
                format = 'rar'
            elif filepath.endswith('.7z'):
                format = '7z'
            else:
                raise ValueError(
                    f"Cannot infer archive format from: {filepath}"
                )

        # Build extracted path
        base_name = os.path.basename(filepath)
        base_name = os.path.splitext(base_name)[0]
        if base_name.endswith('.tar'):
            base_name = os.path.splitext(base_name)[0]
        extracted_path = os.path.join(extract_dir, base_name)

        # Extract based on format
        if format in ['tar', 'tar.gz', 'tgz', 'tar.bz2', 'tbz2']:
            mode_map = {
                'tar': 'r',
                'tar.gz': 'r:gz',
                'tgz': 'r:gz',
                'tar.bz2': 'r:bz2',
                'tbz2': 'r:bz2',
            }
            mode = mode_map[format]
            if not os.path.exists(extracted_path):
                os.makedirs(extracted_path, exist_ok=True)
                print(f"Extracting {filepath} to {extracted_path}")
                with tarfile.open(filepath, mode) as tar:
                    tar.extractall(path=extracted_path)

        elif format == 'zip':
            if not os.path.exists(extracted_path):
                os.makedirs(extracted_path, exist_ok=True)
                print(f"Extracting {filepath} to {extracted_path}")
                with zipfile.ZipFile(filepath, 'r') as zip_ref:
                    zip_ref.extractall(extracted_path)

        elif format == 'gz':
            # Single-file gzip
            output_file = extracted_path
            if not os.path.exists(output_file):
                print(f"Extracting {filepath} to {output_file}")
                with gzip.open(filepath, 'rb') as f_in, open(
                    output_file, 'wb'
                ) as f_out:
                    shutil.copyfileobj(f_in, f_out)
            extracted_path = output_file

        elif format in ['rar', '7z']:
            # Requires external tools unrar / 7z
            if format == 'rar':
                cmd = ['unrar', 'x', filepath, extracted_path]
            else:  # '7z'
                cmd = ['7z', 'x', filepath, f'-o{extracted_path}']

            if not os.path.exists(extracted_path):
                os.makedirs(extracted_path, exist_ok=True)
                print(
                    "Extracting"
                    f" {filepath} to {extracted_path} using {format} extractor"
                )
                try:
                    subprocess.run(cmd, check=True)
                except (subprocess.SubprocessError, FileNotFoundError):
                    raise Exception(
                        f"Failed to extract {format} archive. "
                        "Ensure the correct tool is installed."
                    )
        else:
            raise ValueError(f"Unsupported archive format: {format}")

        # Optionally delete archive
        if delete_archive:
            os.remove(filepath)

        return extracted_path

    def fetch_and_extract(
        self,
        source: str,
        target_dir: Optional[str] = None,
        filename: Optional[str] = None,
        extract: bool = True,
        format: Optional[str] = None,
        delete_archive: bool = False,
        method: str = 'auto',
        force_download: bool = False,
        **kwargs,
    ) -> str:
        """
        Fetch a file and optionally extract it.

        Args:
            source: Source URL or s3/gs path.
            target_dir: Download/extract location.
            filename: Override saved filename.
            extract: Extract the downloaded file.
            format: Archive format (auto-inferred if None).
            delete_archive: Delete archive after extraction.
            method: 'auto', 'http', 'wget', 's3', 'gdown', 'gcs'.
            force_download: Force re-download if file exists.
            **kwargs: Arguments for fetch() and extract().

        Returns:
            Path to extracted directory or file (if extract=False).
        """
        filepath = self.fetch(
            source=source,
            target_dir=target_dir,
            filename=filename,
            method=method,
            force_download=force_download,
            **kwargs,
        )

        # Extract if requested
        if extract:
            return self.extract(
                filepath=filepath,
                extract_dir=target_dir,
                format=format,
                delete_archive=delete_archive,
                **kwargs,
            )
        else:
            return filepath

    def _calculate_noiseceiling(betas, n: int = 1):
        """
        Calculate the noise ceiling from beta estimates.
        Parameters:
            betas: beta estimates in shape (vertices, num_reps, num_stimuli)
            n: A scaling factor (default=1)
        Returns:
            ncsnr: noise SNR at each voxel
            noiseceiling: noise ceiling as a percentage of explainable variance
        """
        import numpy as np  # Ensure numpy is imported
        assert (len(betas.shape) == 3)
        numvertices = betas.shape[0]
        num_reps = betas.shape[1]
        num_vids = betas.shape[2]
        noisesd = np.sqrt(np.mean(np.power(
            np.std(betas, axis=1, keepdims=1, ddof=1), 2), axis=2)).reshape((numvertices,))

        # Calculate the total variance of the single-trial betas.
        totalvar = np.power(
            np.std(np.reshape(betas, (numvertices, num_reps*num_vids)), axis=1), 2)

        # Estimate the signal variance and positively rectify.
        signalvar = totalvar - np.power(noisesd, 2)

        signalvar[signalvar < 0] = 0
        # Compute ncsnr as the ratio between signal std and noise std.
        ncsnr = np.sqrt(signalvar) / noisesd

        # Compute noise ceiling in percentage of explainable variance
        noiseceiling = 100 * (np.power(ncsnr, 2) / (np.power(ncsnr, 2) + 1/n))
        return ncsnr, noiseceiling
