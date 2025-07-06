from typing import List, Dict, Tuple, Set, Optional
import numpy as np
import os
import time
import shutil
import tqdm
import torch
import fasttext
from gensim.models import Word2Vec, KeyedVectors
from sentence_transformers import SentenceTransformer
import logging
import openai
from beir import LoggingHandler
# from FlagEmbedding import FlagICLModel
from collections import Counter
from gensim.parsing.preprocessing import remove_stopwords, preprocess_string
from gensim.utils import simple_preprocess
import hashlib
import gzip

from dotenv import load_dotenv
load_dotenv()

from mistralai import Mistral
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])  # Keep the original handler
logger = logging.getLogger()  # Get the root logger

class EmbeddingGenerator:
    def __init__(self, model_name: str, dataset_name: str, cache_dir: str, device: Optional[torch.device] = None, batch_size: int = 32, force: bool = False):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.pretrained_weights_dir = "./pretrained_weights"
        self.cache_dir = cache_dir
        self.device = device
        self.batch_size = batch_size
        self.force = force
    
    def _get_cache_key(self, texts: List[str], batch_size: int, model_name: str, create_cache_folder: bool = True) -> Tuple[str, str]:
        cache_key_raw = (texts[0][:10] if texts else "") + str(len(texts)) + str(batch_size) + model_name
        cache_key = hashlib.md5(cache_key_raw.encode('utf-8')).hexdigest()
        cache_folder = os.path.join(self.cache_dir, "cache", cache_key)
        if create_cache_folder:
            os.makedirs(cache_folder, exist_ok=True)
        return cache_key, cache_folder
    
    def _generate_batch_embeddings(self, texts: List[str], batch_size: int) -> np.ndarray:
        raise NotImplementedError("You need to implement this method in the subclass.")
    
    def _generate_query_embeddings(self, texts: List[str], batch_size: int) -> np.ndarray:
        # Default implementation is to use the same method as for corpus embeddings
        return self._generate_batch_embeddings(texts, batch_size)
    
    def generate_embeddings(self, text_list: List[str], cache_key: str, type: str = "corpus") -> List[float]:
        if os.path.exists(os.path.join(self.cache_dir, cache_key)):
            embeddings = np.load(os.path.join(self.cache_dir, cache_key))
            logger.info(f"Embeddings for {len(text_list)} texts loaded from {cache_key}.")
            return embeddings.tolist()
        else:
            if type == "corpus":
                embeddings = self._generate_batch_embeddings(text_list, batch_size=self.batch_size)
            elif type == "query":
                embeddings = self._generate_query_embeddings(text_list, batch_size=self.batch_size)

            np.save(os.path.join(self.cache_dir, cache_key), embeddings)
            logger.info(f"Embeddings for {len(text_list)} texts generated and cached to {os.path.join(self.cache_dir, cache_key)}.")
            return embeddings.tolist()

    # can be overridden by subclasses
    def generate_corpus_embeddings(self, corpus: Dict[str, Dict[str, str]], cache_key: str) -> List[float]:
        text_list = [item['text'] for item in corpus.values()]
        return self.generate_embeddings(text_list, cache_key, type="corpus")
    
    # can be overridden by subclasses
    def generate_query_embeddings(self, queries: Dict[str, str], cache_key: str) -> List[float]:
        text_list = [item for item in queries.values()]
        return self.generate_embeddings(text_list, cache_key, type="query")
    

class NVEmbedEmbeddingGenerator(EmbeddingGenerator):
    def __init__(self, cache_dir: str, dataset_name: str, model_settings: dict, force: bool = False):
        device = model_settings.get('device', torch.device('cuda'))
        batch_size = model_settings.get('batch_size', 32)
        super().__init__("NV-Embed", dataset_name, cache_dir, device, batch_size, force)
        model_path = model_settings.get("model_path", "./models/nv-embed/")
        self.model = SentenceTransformer('nvidia/NV-Embed-v2', trust_remote_code=True, cache_folder=model_path)
    
    def _generate_batch_embeddings(self, texts: List[str], batch_size: int) -> np.ndarray:
        total_batches = (len(texts) + batch_size - 1) // batch_size
        cache_key, cache_folder = self._get_cache_key(texts, batch_size, self.model_name)
        
        # Get model's EOS token
        eos_token = self.model.tokenizer.eos_token
        max_length = 512  # Limit text length
        
        # Create final output file
        output_file = os.path.join(cache_folder, "all_embeddings.dat")
        
        
        # Get first batch to determine embedding dimension
        first_batch = texts[:min(batch_size, len(texts))]
        first_batch = [text[:max_length] + eos_token for text in first_batch if text and isinstance(text, str)]
        if not first_batch:
            raise ValueError("No valid texts in the first batch")
        
        # Get embedding dimension
        sample_emb = self.model.encode(
            first_batch[:1],
            batch_size=1,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        embedding_dim = sample_emb.shape[1]

        if os.path.exists(output_file):
            # return np.load(output_file, mmap_mode='r', allow_pickle=True)
            fp = np.memmap(output_file, dtype='float32', mode='r', shape=(len(texts), embedding_dim))
            return fp
        
        # Create memory-mapped file
        fp = np.memmap(output_file, dtype='float32', mode='w+', 
                      shape=(len(texts), embedding_dim))

        
        # Process in batches
        for i in tqdm.tqdm(range(0, len(texts), batch_size), total=total_batches, desc="Generating embeddings"):
            batch_texts = texts[i:i + batch_size]
            # Filter out empty texts and None values
            batch_texts = [text for text in batch_texts if text and isinstance(text, str)]
            if not batch_texts:
                continue
                
            cache_file = os.path.join(cache_folder, f"{i}.npy")
            if os.path.exists(cache_file):
                batch_emb = np.load(cache_file)
            else:
                # Add EOS token to each text and limit length
                batch_texts = [text[:max_length] + eos_token for text in batch_texts]
                
                embeddings = self.model.encode(
                    batch_texts,
                    batch_size=batch_size,
                    normalize_embeddings=True,
                    show_progress_bar=False
                )
                np.save(cache_file, embeddings)
                batch_emb = embeddings
            
            # Write batch_emb directly to memory-mapped file
            fp[i:i + len(batch_emb)] = batch_emb
        
        # Ensure data is written to disk
        fp.flush()
        del fp
        
        # Return memory-mapped array
        return np.memmap(output_file, dtype='float32', mode='r', shape=(len(texts), embedding_dim))

class FastTextEmbeddingGenerator(EmbeddingGenerator):
    def __init__(self, cache_dir: str, dataset_name: str, model_settings: dict, force: bool = False):
        device = model_settings.get('device', torch.device('cuda'))
        batch_size = model_settings.get('batch_size', 32)
        super().__init__("Fast-Text", dataset_name, cache_dir, device, batch_size, force)
        self.model = fasttext.load_model(f"cached_output/fast_text/{self.dataset_name}_fasttext.model")
    
    def _generate_batch_embeddings(self, texts: List[str], batch_size: int) -> np.ndarray:
        texts = [text.replace("\n", " ") for text in texts]
        return np.array([self.model.get_sentence_vector(text) for text in texts])

class Word2VecEmbeddingGenerator(EmbeddingGenerator):
    def __init__(self, cache_dir: str, dataset_name: str, model_settings: dict, force: bool = False):
        device = model_settings.get('device', torch.device('cuda'))
        batch_size = model_settings.get('batch_size', 32)
        super().__init__("Word2Vec", dataset_name, cache_dir, device, batch_size, force)
        
        # Check and download Word2Vec vectors if needed
        model_dir = model_settings.get("model_path", "./models/word2vec/")
        model_file = os.path.join(model_dir, "GoogleNews-vectors-negative300.bin")
        
        if not os.path.exists(model_file):
            self._download_word2vec_vectors(model_dir)
            
        self.model = KeyedVectors.load_word2vec_format(model_file, binary=True)
        self.preprocess = simple_preprocess
        
    def _download_word2vec_vectors(self, model_dir: str) -> None:
        """
        Download Google News Word2Vec vectors.
        
        Args:
            model_dir: Directory to save the vectors
        """
        import requests
        from rich.console import Console
        from rich.progress import Progress, DownloadColumn, BarColumn, TextColumn, TransferSpeedColumn, TimeRemainingColumn
        from rich.prompt import Confirm
        
        console = Console()
        
        # Prompt user for download permission
        console.print(f"\n[bold yellow]🤖 Word2Vec vectors not found![/bold yellow]")
        console.print(f"[blue]Required file:[/blue] GoogleNews-vectors-negative300.bin")
        console.print(f"[blue]Target directory:[/blue] {model_dir}")
        console.print(f"[blue]Download source:[/blue] Google Drive (GoogleNews-vectors-negative300.bin.gz)")
        console.print(f"[blue]File size:[/blue] ~1.5GB")
        console.print(f"[yellow]⚠️  Note: This is a large file and may take some time to download.[/yellow]")
        
        if not Confirm.ask("\n[bold green]Would you like to download Word2Vec vectors automatically?[/bold green]"):
            raise FileNotFoundError(
                f"Word2Vec vectors not found at {model_dir}. "
                f"Please manually download GoogleNews-vectors-negative300.bin.gz from Google Drive "
                f"and extract it to {model_dir}"
            )
        
        # Create directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
        # Note: The original Google Drive link is not direct download friendly
        # We'll use a mirror or alternative source
        download_url = "https://drive.google.com/uc?id=0B7XkCwpI5KDYNlNUTTlSS21pQmM"
        gz_path = os.path.join(model_dir, "GoogleNews-vectors-negative300.bin.gz")
        bin_path = os.path.join(model_dir, "GoogleNews-vectors-negative300.bin")
        
        console.print(f"\n[bold blue]📥 Downloading Word2Vec vectors...[/bold blue]")
        console.print(f"[yellow]⚠️  This download may redirect to Google Drive. If automatic download fails, please download manually.[/yellow]")
        
        # Download with progress bar
        try:
            with Progress(
                TextColumn("[bold blue]Downloading...", justify="right"),
                BarColumn(bar_width=None),
                "[progress.percentage]{task.percentage:>3.1f}%",
                "•",
                DownloadColumn(),
                "•",
                TransferSpeedColumn(),
                "•",
                TimeRemainingColumn(),
                console=console,
            ) as progress:
                
                # Add headers to handle Google Drive
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                }
                
                session = requests.Session()
                response = session.get(download_url, headers=headers, stream=True)
                
                # Handle Google Drive redirect
                if 'drive.google.com' in response.url and 'confirm' in response.url:
                    console.print(f"[yellow]⚠️  Google Drive confirmation required. Please download manually from:[/yellow]")
                    console.print(f"[blue]{download_url}[/blue]")
                    raise FileNotFoundError(
                        f"Automatic download failed due to Google Drive restrictions. "
                        f"Please manually download GoogleNews-vectors-negative300.bin.gz from: {download_url}"
                    )
                
                response.raise_for_status()
                
                total_size = int(response.headers.get('content-length', 0))
                task = progress.add_task("download", total=total_size)
                
                with open(gz_path, 'wb') as file:
                    for chunk in response.iter_content(chunk_size=8192):
                        file.write(chunk)
                        progress.update(task, advance=len(chunk))
            
            console.print(f"[green]✅ Download completed![/green]")
            
        except requests.exceptions.RequestException as e:
            console.print(f"[red]❌ Download failed: {e}[/red]")
            console.print(f"[yellow]💡 Alternative download options:[/yellow]")
            console.print(f"[blue]1. Manual download from:[/blue] {download_url}")
            console.print(f"[blue]2. Use gensim downloader:[/blue] python -c \"import gensim.downloader as api; api.load('word2vec-google-news-300')\"")
            raise FileNotFoundError(
                f"Failed to download Word2Vec vectors. Please download manually from: {download_url}"
            )
        
        # Extract the gz file
        console.print(f"\n[bold blue]📦 Extracting Word2Vec vectors...[/bold blue]")
        
        try:
            with gzip.open(gz_path, 'rb') as gz_file:
                with open(bin_path, 'wb') as bin_file:
                    # Extract with progress
                    with Progress(
                        TextColumn("[bold blue]Extracting...", justify="right"),
                        BarColumn(bar_width=None),
                        console=console,
                    ) as progress:
                        task = progress.add_task("extract", total=None)
                        
                        chunk_size = 8192
                        while True:
                            chunk = gz_file.read(chunk_size)
                            if not chunk:
                                break
                            bin_file.write(chunk)
                            progress.update(task, advance=len(chunk))
            
            console.print(f"[green]✅ Extraction completed![/green]")
            
            # Verify the file exists
            if os.path.exists(bin_path):
                console.print(f"[green]✅ Word2Vec vectors ready at: {bin_path}[/green]")
            else:
                console.print(f"[red]❌ Extracted file not found: {bin_path}[/red]")
                raise FileNotFoundError(f"Expected file not found after extraction: {bin_path}")
            
            # Clean up gz file
            try:
                os.remove(gz_path)
                console.print(f"[dim]🗑️  Cleaned up compressed file[/dim]")
            except OSError:
                console.print(f"[yellow]⚠️  Could not remove compressed file: {gz_path}[/yellow]")
                
        except (gzip.BadGzipFile, OSError) as e:
            console.print(f"[red]❌ Failed to extract file: {e}[/red]")
            # Clean up corrupted file
            try:
                os.remove(gz_path)
            except OSError:
                pass
            raise FileNotFoundError(
                f"Downloaded file appears to be corrupted. Please try again or download manually from: {download_url}"
            )
        
        console.print(f"\n[bold green]🎉 Word2Vec vectors are ready to use![/bold green]")
        
    def _generate_batch_embeddings(self, texts: List[str], batch_size: int) -> np.ndarray:
        embeddings = []
        for text in texts:
            # Use gensim's preprocessing function
            words = self.preprocess(text)
            
            # Calculate document vector
            try:
                doc_vector = self.model.get_mean_vector(
                    words,
                    ignore_missing=True,  # Ignore OOV words
                    # normalized=True  # Return normalized vectors
                )
                embeddings.append(doc_vector)
            except (KeyError, ValueError):  # Handle case where no words are in vocabulary
                embeddings.append(np.zeros(self.model.vector_size))
                
        return np.array(embeddings)

class MistralEmbeddingGenerator(EmbeddingGenerator):
    def __init__(self, cache_dir: str, dataset_name: str, model_settings: dict, force: bool = False):
        device = model_settings.get('device', torch.device('cuda'))
        batch_size = model_settings.get('batch_size', 32)
        super().__init__("Mistral", dataset_name, cache_dir, device, batch_size, force)
        self.api_key = model_settings.get('api_key', str(os.getenv("MISTRAL_API_KEY")))
        self.model_name = model_settings.get('model_name', "mistral-embed")
        self.client = Mistral(api_key=self.api_key)
        self.max_tokens = model_settings.get('max_tokens', 8192)
        self.sleep_time = model_settings.get('sleep_time', 2)

    def _generate_batch_embeddings(self, texts: List[str], batch_size: int) -> np.ndarray:
        import time
        import hashlib
        embeddings = []
        # Calculate global cache key
        cache_key_raw = (texts[0][:10] if texts else "") + str(len(texts)) + str(batch_size)
        cache_key = hashlib.md5(cache_key_raw.encode('utf-8')).hexdigest()
        cache_folder = os.path.join(self.cache_dir, cache_key)
        if self.force:
            if os.path.exists(cache_folder):
                shutil.rmtree(cache_folder)
        os.makedirs(cache_folder, exist_ok=True)

        for i in tqdm.tqdm(range(0, len(texts), batch_size), desc="Generating batch embeddings (Mistral)"):
            batch_texts = texts[i:i + batch_size]
            batch_texts = [text[:self.max_tokens] for text in batch_texts]
            cache_file = os.path.join(cache_folder, f"{i}.npy")
            if os.path.exists(cache_file):
                batch_emb = np.load(cache_file)
                embeddings.append(batch_emb)
                continue
            response = self.client.embeddings.create(
                inputs=[text[:self.max_tokens] for text in batch_texts],
                model=self.model_name
            )
            time.sleep(self.sleep_time)
            batch_emb = np.array([r.embedding for r in response.data])
            np.save(cache_file, batch_emb)
            embeddings.append(batch_emb)
        return np.vstack(embeddings)

class OpenAIEmbeddingGenerator(EmbeddingGenerator):
    def __init__(self, cache_dir: str, dataset_name: str, model_settings: dict, force: bool = False):
        device = model_settings.get('device', torch.device('cuda'))
        batch_size = model_settings.get('batch_size', 32)
        super().__init__("OpenAI", dataset_name, cache_dir, device, batch_size, force)
        self.api_key = model_settings.get('api_key', str(os.getenv("OPENAI_API_KEY")))
        self.model_name = model_settings.get('model_name', "text-embedding-3-small")
        openai.api_key = self.api_key
        self.max_tokens = model_settings.get('max_tokens', 8192 * 4)
    
    def _generate_batch_embeddings(self, texts: List[str], batch_size: int) -> np.ndarray:
        import time
        embeddings = []
        for i in tqdm.tqdm(range(0, len(texts), batch_size), desc="Generating batch embeddings (OpenAI)"):
            batch_texts = texts[i:i + batch_size]
            batch_texts_fill_empty = [text[:self.max_tokens] if text != "" else " " for text in batch_texts]
            response = openai.embeddings.create(input=batch_texts_fill_empty, model=self.model_name)
            time.sleep(3)  # Adjust this time if needed
            embeddings.append(np.array([r.embedding for r in response.data]))
        return np.vstack(embeddings)

class GTEEmbeddingGenerator(EmbeddingGenerator):
    def __init__(self, cache_dir: str, dataset_name: str, model_settings: dict, force: bool = False):
        device = model_settings.get('device', torch.device('cuda'))
        batch_size = model_settings.get('batch_size', 32)
        super().__init__("GTE", dataset_name, cache_dir, device, batch_size, force)
        model_path = model_settings.get("model_path", "./models/gte/")
        self.max_tokens = model_settings.get('max_tokens', 8192)
        self.model_name = model_settings.get('model_name', "Alibaba-NLP/gte-Qwen2-7B-instruct")
        self.model = SentenceTransformer(self.model_name, trust_remote_code=True, cache_folder=model_path)
        self.model.max_seq_length = self.max_tokens

    def _generate_batch_embeddings(self, texts: List[str], batch_size: int) -> np.ndarray:
        return self.model.encode_corpus(texts, batch_size=batch_size, convert_to_numpy=True)
    
    def _generate_query_embeddings(self, texts: List[str], batch_size: int) -> np.ndarray:
        return self.model.encode_queries(texts, batch_size=batch_size, convert_to_numpy=True)

class GloVeEmbeddingGenerator(EmbeddingGenerator):
    def __init__(self, cache_dir: str, dataset_name: str, model_settings: dict, force: bool = False):
        """
        Args:
            dim: Word vector dimension, options are 50, 100, 200, 300
            model_path: Path to the GloVe vectors
            device: Device to use for computation (cuda/cpu)
            batch_size: Batch size for embedding generation
            force: Whether to force re-download of GloVe vectors
            model_name: Name of the GloVe model
            max_tokens: Maximum number of tokens to embed
            sleep_time: Time to sleep between requests

        """
        device = model_settings.get('device', torch.device('cuda'))
        batch_size = model_settings.get('batch_size', 32)
        dim = model_settings.get('dim', 300)
        super().__init__("GloVe", dataset_name, cache_dir, device, batch_size, force)
        if dim not in [50, 100, 200, 300]:
            raise ValueError("GloVe dimension must be one of: 50, 100, 200, 300")
            
        # Check and download GloVe vectors if needed
        model_dir = model_settings.get("model_path", "./models/glove.6B/")
        model_file = os.path.join(model_dir, f"glove.6B.{dim}d.txt")
        
        if not os.path.exists(model_file):
            self._download_glove_vectors(model_dir, dim)
        
        # Load GloVe vectors
        self.model = {}
        logger.info(f"Loading GloVe {dim}d vectors from {model_file}...")
        with open(model_file, 'r', encoding='utf-8') as f:
            for line in tqdm.tqdm(f, desc=f"Loading GloVe {dim}d vectors"):
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], dtype='float32')
                self.model[word] = vector
        
        self.vector_size = dim
        self.preprocess = simple_preprocess
        logger.info(f"Loaded {len(self.model)} word vectors of dimension {dim}")
    
    def _download_glove_vectors(self, model_dir: str, dim: int) -> None:
        """
        Download and extract GloVe vectors from Stanford NLP.
        
        Args:
            model_dir: Directory to save the vectors
            dim: Dimension of the vectors (50, 100, 200, 300)
        """
        import requests
        import zipfile
        from rich.console import Console
        from rich.progress import Progress, DownloadColumn, BarColumn, TextColumn, TransferSpeedColumn, TimeRemainingColumn
        from rich.prompt import Confirm
        
        console = Console()
        
        # Prompt user for download permission
        console.print(f"\n[bold yellow]🤖 GloVe vectors not found![/bold yellow]")
        console.print(f"[blue]Required file:[/blue] glove.6B.{dim}d.txt")
        console.print(f"[blue]Target directory:[/blue] {model_dir}")
        console.print(f"[blue]Download source:[/blue] https://nlp.stanford.edu/data/glove.6B.zip")
        console.print(f"[blue]File size:[/blue] ~822MB")
        
        if not Confirm.ask("\n[bold green]Would you like to download GloVe vectors automatically?[/bold green]"):
            raise FileNotFoundError(
                f"GloVe vectors not found at {model_dir}. "
                f"Please manually download glove.6B.zip from https://nlp.stanford.edu/data/glove.6B.zip "
                f"and extract it to {model_dir}"
            )
        
        # Create directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
        # Download URL
        download_url = "https://nlp.stanford.edu/data/glove.6B.zip"
        zip_path = os.path.join(model_dir, "glove.6B.zip")
        
        console.print(f"\n[bold blue]📥 Downloading GloVe vectors...[/bold blue]")
        
        # Download with progress bar
        try:
            with Progress(
                TextColumn("[bold blue]Downloading...", justify="right"),
                BarColumn(bar_width=None),
                "[progress.percentage]{task.percentage:>3.1f}%",
                "•",
                DownloadColumn(),
                "•",
                TransferSpeedColumn(),
                "•",
                TimeRemainingColumn(),
                console=console,
            ) as progress:
                
                response = requests.get(download_url, stream=True)
                response.raise_for_status()
                
                total_size = int(response.headers.get('content-length', 0))
                task = progress.add_task("download", total=total_size)
                
                with open(zip_path, 'wb') as file:
                    for chunk in response.iter_content(chunk_size=8192):
                        file.write(chunk)
                        progress.update(task, advance=len(chunk))
            
            console.print(f"[green]✅ Download completed![/green]")
            
        except requests.exceptions.RequestException as e:
            console.print(f"[red]❌ Download failed: {e}[/red]")
            raise FileNotFoundError(
                f"Failed to download GloVe vectors. Please manually download from: {download_url}"
            )
        
        # Extract the zip file
        console.print(f"\n[bold blue]📦 Extracting GloVe vectors...[/bold blue]")
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # List files in the zip
                file_list = zip_ref.namelist()
                console.print(f"[blue]Files in archive:[/blue] {len(file_list)} files")
                
                # Extract all files with progress
                with Progress(
                    TextColumn("[bold blue]Extracting...", justify="right"),
                    BarColumn(bar_width=None),
                    "[progress.percentage]{task.percentage:>3.1f}%",
                    console=console,
                ) as progress:
                    task = progress.add_task("extract", total=len(file_list))
                    
                    for file_info in file_list:
                        zip_ref.extract(file_info, model_dir)
                        progress.update(task, advance=1)
            
            console.print(f"[green]✅ Extraction completed![/green]")
            
            # Verify the required file exists
            required_file = os.path.join(model_dir, f"glove.6B.{dim}d.txt")
            if os.path.exists(required_file):
                console.print(f"[green]✅ GloVe {dim}d vectors ready at: {required_file}[/green]")
            else:
                console.print(f"[red]❌ Required file not found: {required_file}[/red]")
                raise FileNotFoundError(f"Expected file not found after extraction: {required_file}")
            
            # Clean up zip file
            try:
                os.remove(zip_path)
                console.print(f"[dim]🗑️  Cleaned up zip file[/dim]")
            except OSError:
                console.print(f"[yellow]⚠️  Could not remove zip file: {zip_path}[/yellow]")
                
        except zipfile.BadZipFile as e:
            console.print(f"[red]❌ Failed to extract zip file: {e}[/red]")
            # Clean up corrupted zip file
            try:
                os.remove(zip_path)
            except OSError:
                pass
            raise FileNotFoundError(
                f"Downloaded file appears to be corrupted. Please try again or download manually from: {download_url}"
            )
        
        console.print(f"\n[bold green]🎉 GloVe vectors are ready to use![/bold green]")
        
    def _generate_batch_embeddings(self, texts: List[str], batch_size: int) -> np.ndarray:
        embeddings = []
        for text in texts:
            # Use gensim's preprocessing function for tokenization
            words = self.preprocess(text)
            
            # Collect found word vectors
            vectors = []
            for word in words:
                if word in self.model:
                    vectors.append(self.model[word])
            
            if vectors:
                # If word vectors are found, take the average
                doc_vector = np.mean(vectors, axis=0)
            else:
                # If no word vectors are found, return zero vector
                doc_vector = np.zeros(self.vector_size)
            
            embeddings.append(doc_vector)
                
        return np.array(embeddings)

def get_embedding_generator(model_name: str, dataset_name: str, cache_dir: str, model_settings: dict, force: bool = False):
    if "model_path" in model_settings:
        model_settings["model_path"] = os.path.expanduser(model_settings["model_path"])
    if model_name.lower() == "nv-embed":
        return NVEmbedEmbeddingGenerator(cache_dir, dataset_name, model_settings, force)
    elif model_name.lower() == "fast-text":
        return FastTextEmbeddingGenerator(cache_dir, dataset_name, model_settings, force)
    elif model_name.lower() == "word2vec":
        return Word2VecEmbeddingGenerator(cache_dir, dataset_name, model_settings, force)
    elif model_name.lower() == "mistral":
        return MistralEmbeddingGenerator(cache_dir, dataset_name, model_settings, force)
    elif model_name.lower() == "openai" or model_name.lower() == "gpt3":
        return OpenAIEmbeddingGenerator(cache_dir, dataset_name, model_settings, force)
    elif model_name.lower() == "gte":
        return GTEEmbeddingGenerator(cache_dir, dataset_name, model_settings, force)
    elif model_name.lower() == "glove":
        return GloVeEmbeddingGenerator(cache_dir, dataset_name, model_settings, force)
    else:
        raise ValueError(f"Invalid model name: {model_name}")
