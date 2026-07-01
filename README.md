# RAG sistem sa semantickom pretragom

Ovaj projekat implementira RAG (Retrieval-Augmented Generation) sistem koji odgovara
na pitanja na osnovu lokalnih PDF i TXT dokumenata. Sistem prvo ucitava dokumente,
dijeli ih na manje dijelove, pretvara tekst u embedding vektore, indeksira ih u
vector store i zatim pronalazi najrelevantnije dijelove dokumentacije za dato pitanje.

## Tema projekta

** RAG sistem sa semantickom pretragom**

Zadatak:

- Napraviti sistem koji odgovara na pitanja na osnovu lokalnih PDF ili TXT dokumenata.
- Koristiti embedding modele.
- Indeksirati dokumente u vector store.
- Uporediti cosine similarity i Euclidean distance.

## Funkcionalnosti

- Ucitavanje lokalnih `.txt` i `.pdf` dokumenata iz foldera `data/documents`.
- Dijeljenje dokumenata na chunkove radi preciznije pretrage.
- Kreiranje embedding vektora za svaki chunk.
- Indeksiranje embeddinga u vector store.
- Semanticka pretraga po pitanju korisnika.
- Generisanje odgovora na osnovu najrelevantnijih pronadjenih dijelova teksta.
- Poredjenje rezultata pomocu:
  - cosine similarity
  - Euclidean distance
- Podrska za dva embedding pristupa:
  - lokalni hashing embedding model
  - `sentence-transformers` model
- Podrska za dva vector store backend-a:
  - `numpy`
  - `faiss`

## Struktura projekta

```text
AI-projekat/
+-- data/
|   +-- documents/          # Lokalni TXT/PDF dokumenti
+-- rag/
|   +-- chunking.py         # Dijeljenje dokumenata na chunkove
|   +-- document_loader.py  # Ucitavanje TXT i PDF fajlova
|   +-- embeddings.py       # Embedding modeli
|   +-- models.py           # Pomocni modeli podataka
|   +-- rag_pipeline.py     # Glavni RAG tok
|   +-- response.py         # Formatiranje odgovora
|   +-- vector_store.py     # Numpy/FAISS vector store
+-- tests/                  # Testovi
+-- main.py                 # CLI aplikacija
+-- requirements.txt        # Python zavisnosti
+-- README.md
```

## Kako sistem radi

1. Dokumenti se ucitavaju iz foldera `data/documents`.
2. Tekst se dijeli na manje chunkove.
3. Svaki chunk se pretvara u embedding vektor.
4. Embedding vektori se spremaju u vector store.
5. Korisnicko pitanje se takodje pretvara u embedding.
6. Sistem poredi embedding pitanja sa embedding vektorima dokumenata.
7. Najslicniji chunkovi se koriste za formiranje odgovora.

## Instalacija

Preporuceno je koristiti virtuelno okruzenje:

```bash
python -m venv .venv
```

Aktivacija na Windowsu:

```bash
.venv\Scripts\activate
```

Instalacija zavisnosti:

```bash
pip install -r requirements.txt
```

## Pokretanje

Pokretanje interaktivnog moda:

```bash
python main.py
```

Postavljanje jednog pitanja:

```bash
python main.py --question "Sta je RAG sistem?"
```

Pokretanje demo pitanja:

```bash
python main.py --demo
```

Poredjenje cosine similarity i Euclidean distance:

```bash
python main.py --demo --compare

python main.py --docs data/documents --question "Sta je RAG sistem?" --compare 
```


Koristenje FAISS vector store-a:

```bash
python main.py --vector-store faiss --demo
```

Koristenje SentenceTransformer embedding modela:

```bash
python main.py --embedding sentence-transformer --demo
```

Primjer sa svim opcijama:

```bash
python main.py --docs data/documents --embedding local --vector-store faiss --metric cosine --top-k 3 --question "Kako se koristi cosine similarity?"
```

## CLI opcije

| Opcija | Opis |
| --- | --- |
| `--docs` | Folder sa lokalnim TXT/PDF dokumentima. |
| `--question` | Pitanje koje se postavlja sistemu. |
| `--demo` | Pokrece unaprijed definisana demo pitanja. |
| `--compare` | Uporedjuje rezultate za cosine i Euclidean metriku. |
| `--top-k` | Broj najrelevantnijih rezultata koji se prikazuju. |
| `--embedding` | Izbor embedding modela: `local` ili `sentence-transformer`. |
| `--model` | Naziv konkretnog SentenceTransformer modela. |
| `--vector-store` | Izbor vector store backend-a: `numpy` ili `faiss`. |
| `--metric` | Metrika pretrage: `cosine` ili `euclidean`. |

## Cosine similarity i Euclidean distance

U projektu se porede dvije metrike za semanticku pretragu:

### Cosine similarity

Cosine similarity mjeri ugaonu slicnost izmedju dva vektora. Vrijednost je veca kada
su embedding vektori semanticki slicniji. Ova metrika je cesto pogodna za tekstualne
embeddinge jer se fokusira na smjer vektora, a ne samo na njegovu duzinu.

### Euclidean distance

Euclidean distance mjeri direktnu udaljenost izmedju dva vektora u vektorskom
prostoru. Manja vrijednost znaci vecu slicnost. Korisna je za poredjenje kada je
bitna geometrijska udaljenost embeddinga.

### Poredjenje u aplikaciji

Komanda:

```bash
python main.py --question "Koja je razlika izmedju fine-tuninga i RAG-a?" --compare
```

ispisuje rezultate za obje metrike, pa se moze vidjeti da li sistem pronalazi iste
ili razlicite chunkove kao najrelevantnije.

## Lokalni dokumenti

Dokumenti koji se indeksiraju nalaze se u:

```text
data/documents
```

U taj folder se mogu dodati novi `.txt` ili `.pdf` fajlovi. Nakon dodavanja novih
dokumenata dovoljno je ponovo pokrenuti program, jer se indeks kreira pri pokretanju.

## Testiranje

Ako su instalirane sve zavisnosti, testovi se mogu pokrenuti komandom:

```bash
pytest
```

## Zakljucak

Projekat prikazuje osnovni RAG tok nad lokalnim dokumentima: ucitavanje dokumenata,
kreiranje embeddinga, indeksiranje u vector store, semanticku pretragu i poredjenje
cosine similarity i Euclidean distance metrika.
