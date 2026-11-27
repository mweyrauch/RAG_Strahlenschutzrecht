# RAG für das Strahlenschutzrecht

Ein kleiner, nachvollziehbarer Prototyp für Retrieval-Augmented Generation (RAG) auf dem
Strahlenschutzgesetz (StrlSchG) und der Strahlenschutzverordnung (StrlSchV). Der Schwerpunkt liegt
auf der Auswertung: Wie gut findet das System die richtige Vorschrift, wie gut sind die Antworten,
und wo macht es Fehler?

> Keine Rechtsberatung. Antworten des Systems können falsch oder unvollständig sein.

**Aufbau in einem Satz:** Die Vorschriften werden in Absätze zerlegt, mit mehrsprachigen Embeddings
(`intfloat/multilingual-e5-large`) durchsucht, und ein lokales Sprachmodell (`qwen2.5:7b` über
Ollama) beantwortet die Frage aus den fünf besten Absätzen mit Quellenangaben. Als Vergleich dienen
die wortbasierte Suche BM25 und eine extraktive Antwort ohne Sprachmodell.

**Fragenset:** 40 selbst erstellte Items, jedes in zwei Formen: **fachsprachlich**, nah am
Gesetzeswortlaut, und **alltagssprachlich**, ohne die Begriffe der Vorschrift. Dazu 15 Fragen, die
der Korpus nicht beantwortet (Fremdfragen). In Klammern stehen 95-%-Bootstrap-Intervalle über Items.

## Ergebnisse

### Suche

| Retriever | Frageform | Recall@1 | Recall@5 | MRR |
|---|---|---|---|---|
| BM25 | fachsprachlich | 0,80 | 0,95 [0,88; 1,00] | 0,86 [0,77; 0,94] |
| BM25 | alltagssprachlich | 0,00 | 0,10 [0,03; 0,20] | 0,05 [0,02; 0,08] |
| **e5-large** | fachsprachlich | 0,88 | 1,00 [1,00; 1,00] | 0,93 [0,88; 0,99] |
| **e5-large** | alltagssprachlich | 0,43 | **0,78** [0,65; 0,90] | **0,57** [0,44; 0,69] |

*Recall@k:* Anteil der Fragen mit einer richtigen Vorschrift unter den ersten k Treffern.
*MRR:* Mittelwert von 1/Rang der ersten richtigen Vorschrift (0, wenn nicht unter den ersten 50).

- **Embeddings schließen den größten Teil der Sprachlücke.** Bei Alltagsfragen steigt Recall@5
  gegenüber BM25 um 0,68 [0,50; 0,85] (gepaarte Differenz). Bei Fachfragen sind beide ähnlich gut.
- **Einzelne Begriffe bleiben schwierig.** Die Frage nach der „Sicherheitsbelehrung“ findet § 63
  StrlSchV („Unterweisung“) erst auf Platz 17.
- **Größere Modelle helfen deutlich.** e5-large wurde gegen kleinere Varianten derselben Familie
  ausgewählt, auf denselben 40 Items:

| Modell | Größe | Recall@5 Alltag | MRR Fach | AUC Fremdfragen |
|---|---|---|---|---|
| BM25 | – | 0,10 | 0,86 | 0,80 |
| e5-small | ca. 470 MB | 0,45 | 0,86 | 0,98 |
| e5-base | ca. 1,1 GB | 0,63 | 0,87 | 0,97 |
| **e5-large** | ca. 2,2 GB | **0,78** | **0,93** | 0,97 |

Die AUC gibt an, wie gut der Ähnlichkeitswert des besten Treffers Korpus- von Fremdfragen trennt.
Eine feste Schwelle darauf ist trotzdem unzuverlässig: Bei e5-large passieren 5 der 15 Fremdfragen
eine Schwelle, die 95 % der Korpusfragen durchlässt, darunter Feuerlöscher und Steuererklärung.

![Recall@k](docs/abbildungen/recall_kurven.png)

### Antworten

Eine Antwort gilt automatisch als korrekt, wenn das System sich nicht enthält und ein vorab
festgelegtes Antwortmuster im Text steht (z. B. „400 Millisievert“). Bei Fremdfragen ist die
Enthaltung korrekt. Abruf jeweils mit e5-large:

| Antwort | Fachfragen korrekt | Alltagsfragen korrekt | Fremdfragen enthalten |
|---|---|---|---|
| extraktiv (Satzauswahl per TF-IDF) | 0,68 [0,53; 0,80] | 0,20 [0,08; 0,33] | 5 von 15 |
| **qwen2.5:7b** | **0,90** [0,80; 0,98] | **0,48** [0,33; 0,63] | **15 von 15** |

- **Das Sprachmodell nutzt die Quellen deutlich besser.** Es erkennt etwa, dass „die Summe der in
  allen Kalenderjahren ermittelten effektiven Dosen“ (§ 77 StrlSchG) die Frage nach der Strahlung
  „über das ganze Arbeitsleben“ beantwortet. Die extraktive Antwort findet diesen Satz nicht.
- **Zitate sind verlässlich.** Jede korrekt gewertete Antwort zitiert eine Soll-Vorschrift;
  Zitate auf nicht vorhandene Quellen kamen nicht vor.
- **Das Modell enthält sich zuverlässiger als eine Score-Schwelle**, bei allen 15 Fremdfragen.

**Manuelle Prüfung.** Die 25 nicht korrekt gewerteten Antworten wurden einzeln mit dem Gesetzestext
verglichen (`data/eval/pruefung_llm.yaml`):

| Einordnung | Anzahl | Beispiel |
|---|---|---|
| inhaltlich richtig, Muster zu eng oder Frage mehrdeutig | 9 | q24: nennt 250 und 500 mSv aus § 114 Abs. 3, das Muster kannte nur 100 mSv aus Abs. 2 |
| unvollständig | 5 | q27: nennt das Bundesamt für Strahlenschutz, nicht aber Aufsichts- und Polizeibehörde |
| falsch | 3 | q36 (unten), q40: bestreitet eine zentrale Speicherung der Dosisdaten |
| enthalten | 8 | in 4 Fällen stand die Antwort in den Quellen |

Inhaltlich richtig sind damit 64 von 80 Antworten (automatisch: 55); 3 von 72 beantworteten Fragen
enthalten eine falsche Aussage. Die Einordnung ist von Hand vorgenommen und nicht unabhängig
geprüft; die Muster wurden nach dem Lauf nicht angepasst.

**Ein sicherheitsrelevanter Fehler.** Auf „Ich stille noch. Darf ich weiter im Isotopenlabor
arbeiten?“ antwortet das Modell „Ja, Sie dürfen weiter im Isotopenlabor arbeiten …“. § 69 StrlSchV
verlangt dagegen, dass bei stillenden Personen eine innere berufliche Exposition ausgeschlossen ist.
§ 69 war nicht unter den Quellen; statt sich zu enthalten, schließt das Modell aus verwandten
Vorschriften. Antworten zu Schutzpflichten brauchen daher eine fachliche Prüfung, auch wenn sie
Quellen zitieren.

### Verworfen: Umformulierung der Anfrage

Ein Sprachmodell sollte Alltagsfragen vor der Suche in Fachsprache übersetzen. Bei Einzeltests mit
`qwen2.5:3b` und `qwen2.5:7b` blieb „Belehrung“ statt „Unterweisung“ stehen, die Bedeutung
verschob sich, und das 7B-Modell erfand eine Fundstelle. Der Ansatz wurde nicht weiter verfolgt.

## Schnellstart

Auf Linux ohne GPU zuerst die kleinere CPU-Version von PyTorch installieren:

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev,dense]"
pytest -q                                          # ohne Rohdaten und ohne Modelle
python scripts/lade_korpus.py                      # StrlSchG + StrlSchV nach data/roh/
python scripts/pruefe_fragen.py                    # Fragenset gegen den Wortlaut prüfen
python scripts/experimente.py configs/retrieval.yaml
python scripts/frage.py "Wie oft ist die Unterweisung zu wiederholen?"
```

Mit Sprachmodell (z. B. [Ollama](https://ollama.com) mit `ollama pull qwen2.5:7b`):

```bash
export LLM_BASE_URL=http://localhost:11434/v1 LLM_MODEL=qwen2.5:7b
python scripts/frage.py --backend llm "Wie oft muss ich zur Strahlenschutz-Belehrung?"
python scripts/experimente.py configs/antworten.yaml
```

Ergebnisse landen in `ergebnisse/retrieval/` bzw. `ergebnisse/antworten/`. Embeddings und Antworten
des Sprachmodells werden in `data/cache/` zwischengespeichert; ein abgebrochener Lauf setzt dort
wieder an. Auf einem Laptop ohne GPU (16 GB RAM) brauchte qwen2.5:7b etwa 100 Sekunden je Frage.

## Aufbau

```
src/strlrag/
  korpus.py      XML von gesetze-im-internet.de -> Vorschriften mit Titel und Absätzen
  text.py        Normalisierung, Stoppwörter, CISTEM-Stemmer, Satzzerlegung
  chunking.py    ein Absatz je Chunk, Titel der Vorschrift als Präfix
  retrieval.py   Embeddings (e5), BM25
  antwort.py     Prompt mit nummerierten Quellen, extraktive Antwort, Sprachmodell-Anbindung,
                 Erkennung der Enthaltung, Prüfung der Zitate
  evaluation.py  Rang auf Paragraphenebene, Recall@k, MRR, Cluster-Bootstrap, AUC
scripts/         lade_korpus.py, pruefe_fragen.py, experimente.py, frage.py, abbildungen.py
configs/         retrieval.yaml, antworten.yaml
data/eval/       fragen.yaml (Fragenset), pruefung_llm.yaml (manuelle Prüfung)
tests/           48 Tests auf einem synthetischen Testgesetz
```

## Methode

- **Korpus:** XML-Fassungen von [gesetze-im-internet.de](https://www.gesetze-im-internet.de)
  (StrlSchG Stand 02.11.2026, StrlSchV Stand 05.08.2025), 459 Vorschriften mit 1.397 Absätzen.
- **Zerlegung:** ein Absatz je Chunk, mit dem Titel der Vorschrift als Präfix, weil viele Absätze
  für sich nicht sagen, worum es geht. Absätze erlauben präzise Quellenangaben („§ 78 Abs. 4“).
- **Retriever:** e5 mit den Präfixen „query: “ und „passage: “, Kosinus-Ähnlichkeit. BM25
  (k1 = 1,2, b = 0,75) auf Tokens mit gefalteten Umlauten und CISTEM-Stemming.
- **Antwort:** Der Systemprompt verlangt, nur die nummerierten Quellen zu nutzen, jede Aussage mit
  [n] zu belegen und sich sonst mit einer festen Formel zu enthalten. Temperatur 0, jede Quelle im
  Prompt auf 250 Wörter gekürzt, damit sie in das Kontextfenster kleiner Modelle passt.
- **Bewertung:** Ein Treffer zählt auf Paragraphenebene, mehrere Absätze derselben Vorschrift
  zählen einmal. Bootstrap über Items mit 2.000 Ziehungen, Vergleiche gepaart.
- **Fragenset:** Je Item Soll-Vorschriften, ein wörtliches Stichwort und kurze Antwortmuster.
  `pruefe_fragen.py` prüft, dass Stichwort und mindestens ein Muster in der Soll-Vorschrift stehen;
  nach einer Gesetzesänderung sollte es mit frisch geladenen Texten erneut laufen.

## Grenzen

- **Kleines, verzerrtes Fragenset.** 40 Items von einer Person. Fachfragen sind für wortbasierte
  Suche eher zu leicht, Alltagsfragen eher zu schwer. Die Intervalle sind entsprechend breit.
- **Auswahl auf denselben Fragen.** Embedding-Modell und Schwelle wurden an denselben Fragen gewählt,
  an denen sie gemessen werden; die Werte sind dadurch eher etwas zu optimistisch.
- **Grobe Antwortbewertung.** Die Muster übersehen korrekte, anders formulierte Antworten und
  erkennen keine falschen Zusatzaussagen in sonst korrekt gewerteten Antworten.
- **Ein Sprachmodell, ein Lauf.** Andere Modelle können sich deutlich anders verhalten.
- **Lange Absätze** (vor allem Tabellen in Anlagen) werden von e5 nach 512 Tokens abgeschnitten.

## Nächste Schritte

1. Sicherheitsrelevante Themen (Schwangerschaft, Stillzeit, Jugendliche) gezielt testen und
   Antworten dazu nur mit Hinweis auf fachliche Prüfung ausgeben.
2. Weitere Sprachmodelle vergleichen, besonders bei den falschen Antworten und den
   übervorsichtigen Enthaltungen.
3. Das Fragenset von weiteren Personen formulieren lassen und einen Teil zum Festlegen von Modell
   und Schwelle zurückhalten.
4. Die extraktive Basislinie stärken: Bei q02 steht „Berufslebensdosis“ nur im Titel von § 77,
   nicht im Satz mit dem Grenzwert; der Titel sollte beim Satzvergleich mitzählen.

## Lizenz

Der Code steht unter der [MIT-Lizenz](LICENSE). Die Gesetzestexte sind nicht im Repository
enthalten; `scripts/lade_korpus.py` lädt sie von gesetze-im-internet.de. Als amtliche Werke
(§ 5 UrhG) sind sie urheberrechtlich frei.
