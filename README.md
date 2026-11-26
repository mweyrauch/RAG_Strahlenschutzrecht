# RAG für das Strahlenschutzrecht

Ein kleiner, nachvollziehbarer Prototyp für Retrieval-Augmented Generation (RAG) auf dem
Strahlenschutzgesetz (StrlSchG) und der Strahlenschutzverordnung (StrlSchV), mit einer
Auswertung, die die Schwächen des Systems sichtbar macht statt sie zu verdecken.

Das Projekt hat drei Teile: Abschnitte des Korpus abrufen, daraus eine Antwort mit Quellenangaben bauen und
messen, wie gut beides funktioniert. Der Schwerpunkt liegt auf dem dritten Teil.

> Keine Rechtsberatung. Antworten des Systems können falsch oder unvollständig sein.

## Kernergebnisse

Abgerufen wird mit mehrsprachigen Embeddings (`intfloat/multilingual-e5-large`) auf Absätzen,
jeweils mit dem Titel der Vorschrift als Präfix. Die Antwort formuliert ein lokales Sprachmodell
(`qwen2.5:7b` über Ollama) aus den fünf besten Absätzen. Zum Vergleich dienen die wortbasierte
Suche BM25 mit Stemming und eine extraktive Antwort ohne Sprachmodell. Gemessen wurde auf 40 selbst erstellten Fragen-Items. Jedes Item gibt es in zwei
Formen: einmal **fachsprachlich**, nah am Gesetzeswortlaut, und einmal **alltagssprachlich**, ohne die
Begriffe der Vorschrift. In Klammern stehen 95-%-Bootstrap-Intervalle über Items.

| Retriever | Frageform | Recall@1 | Recall@5 | MRR |
|---|---|---|---|---|
| BM25 | fachsprachlich | 0,80 | 0,95 [0,88; 1,00] | 0,86 [0,77; 0,94] |
| BM25 | alltagssprachlich | 0,00 | 0,10 [0,03; 0,20] | 0,05 [0,02; 0,08] |
| **Embeddings** | fachsprachlich | 0,88 | 1,00 [1,00; 1,00] | 0,93 [0,88; 0,99] |
| **Embeddings** | alltagssprachlich | 0,43 | **0,78** [0,65; 0,90] | **0,57** [0,44; 0,69] |

*Recall@k:* Anteil der Fragen, bei denen eine richtige Vorschrift unter den ersten k Treffern ist.
*MRR (Mean Reciprocal Rank):* Mittelwert von 1/Rang der ersten richtigen Vorschrift, also 1 für
Platz 1, 0,5 für Platz 2 und 0, wenn sie nicht unter den ersten 50 Treffern ist.

1. **Embeddings schließen den größten Teil der Sprachlücke.** Bei Alltagsfragen steigt Recall@5
   gegenüber BM25 um 0,68 [0,50; 0,85] und die MRR um 0,52 [0,38; 0,65] (gepaarte Differenzen).
   Unter den ersten 20 Treffern ist die richtige Vorschrift bei Alltagsfragen in 98 % der Fälle,
   bei BM25 in 25 %. Bei Fachfragen sind Embeddings mindestens gleich gut
   (MRR-Differenz +0,07 [−0,02; +0,17]).
2. **Einzelne Begriffe bleiben schwierig.** „Wie oft muss ich zur Sicherheitsbelehrung, wenn ich
   mit radioaktiven Quellen arbeite?“ findet § 63 StrlSchV („Unterweisung“) erst auf Platz 17.
   „Belehrung“ und „Unterweisung“ liegen für alle getesteten Modelle weit auseinander.
3. **Fachfremde Fragen lassen sich über die Ähnlichkeit weitgehend erkennen.** Der Score des besten
   Treffers trennt Korpusfragen von 15 fachfremden Fragen mit AUC = 0,97 [0,93; 0,99], bei BM25
   mit 0,80 [0,68; 0,90]. Bei einer Schwelle, die 95 % der Korpusfragen durchlässt, kommen 5 der 15
   Fremdfragen durch, bei BM25 sind es 11. Welche das sind, steht in `ergebnisse/fremdfragen.csv`.
   Die Schwelle ist allerdings auf denselben Fragen gewählt.
4. **Ein Sprachmodell macht aus den Quellen deutlich bessere Antworten.** Bei Alltagsfragen steht
   die gesuchte Angabe in 75 % der Fälle in den fünf abgerufenen Absätzen. Die extraktive Antwort
   trifft sie nur in 20 % der Fälle, qwen2.5:7b in 48 % (automatische Bewertung, siehe
   [Antworten](#antworten)). Eine von 72 beantworteten Fragen enthielt einen sicherheitsrelevanten
   Fehler.

![Recall@k](docs/abbildungen/recall_kurven.png)

### Antworten

Eine Antwort auf eine Korpusfrage gilt automatisch als korrekt, wenn das System sich nicht
enthält und eines der vorab festgelegten Antwortmuster im Text steht (z. B. „400 Millisievert“).
Bei Fremdfragen ist die Enthaltung korrekt. Abruf jeweils mit e5-large, ohne Score-Schwelle:

| Antwort | Fachfragen korrekt | Alltagsfragen korrekt | Fremdfragen enthalten |
|---|---|---|---|
| extraktiv (Satzauswahl per TF-IDF) | 0,68 [0,53; 0,80] | 0,20 [0,08; 0,33] | 5 von 15 |
| **qwen2.5:7b** | **0,90** [0,80; 0,98] | **0,48** [0,33; 0,63] | **15 von 15** |

1. **Das Sprachmodell nutzt die Quellen deutlich besser.** Es erkennt zum Beispiel, dass „die
   Summe der in allen Kalenderjahren ermittelten effektiven Dosen“ (§ 77 StrlSchG) die Frage nach
   der Strahlung „über das ganze Arbeitsleben“ beantwortet. Die extraktive Antwort findet diesen
   Satz nicht, weil er kein Wort mit der Frage teilt.
2. **Zitate sind verlässlich.** Bei allen korrekt gewerteten Antworten zitiert das Modell eine
   Soll-Vorschrift. Zitate auf nicht vorhandene Quellennummern kamen nicht vor.
3. **Die Enthaltung übernimmt das Modell besser als eine Score-Schwelle.** Alle 15 Fremdfragen
   beantwortet es mit der Enthaltungsformel, auch die 5, die die Schwelle passieren. Eine
   zusätzliche Schwelle verschlechtert dagegen die Alltagsfragen (0,40 statt 0,48), weil sie
   auch beantwortbare Fragen abweist.

**Manuelle Prüfung.** Die 25 Korpusfragen, die die automatische Bewertung nicht als korrekt wertet,
wurden einzeln mit dem Gesetzestext verglichen (`data/eval/pruefung_llm.yaml`):

| Einordnung | Anzahl | Beispiel |
|---|---|---|
| inhaltlich richtig, Muster zu eng oder Frage mehrdeutig | 9 | q24: nennt 250 und 500 mSv aus § 114 Abs. 3, das Muster kannte nur 100 mSv aus Abs. 2 |
| unvollständig | 5 | q27: nennt das Bundesamt für Strahlenschutz, nicht aber Aufsichts- und Polizeibehörde |
| falsch | 3 | q36, q40, q25 (siehe unten) |
| enthalten | 8 | in 4 Fällen stand die Antwort in den Quellen, in 4 Fällen nicht |

Von den 72 beantworteten Korpusfragen enthalten damit 3 eine falsche Aussage. Inhaltlich richtig
sind nach der manuellen Prüfung 64 von 80 Antworten, nach der automatischen Bewertung 55 von 80.
Die Einordnung ist von Hand vorgenommen und nicht unabhängig geprüft. Die Antwortmuster wurden
nach dem Lauf nicht angepasst.

**Ein sicherheitsrelevanter Fehler.** Auf die Alltagsfrage „Ich stille noch. Darf ich weiter im
Isotopenlabor arbeiten?“ antwortet das Modell: „Ja, Sie dürfen weiter im Isotopenlabor arbeiten
…“. § 69 StrlSchV verlangt dagegen, dass bei stillenden Personen eine innere berufliche Exposition
ausgeschlossen ist. § 69 war nicht unter den fünf Quellen. Statt sich zu enthalten, zieht das
Modell aus anderen, nur verwandten Quellen einen falschen Schluss.
Für einen Einsatz im Strahlenschutz heißt das: Antworten zu Schutzpflichten brauchen eine
fachliche Prüfung, auch wenn Zitate vorhanden und formal korrekt sind.

Auf dem Laptop des Autors (CPU, 16 GB RAM) brauchte das Modell im Mittel etwa 100 Sekunden je
Frage. Mit dem Embedding-Modell gleichzeitig im Speicher wurde Ollama zunächst wegen Speichermangels
beendet; seither ruft das Skript alle Quellen vorab ab und gibt das Embedding-Modell vor den
Anfragen frei.

### Wahl des Embedding-Modells

Verglichen wurden drei Größen derselben Modellfamilie (`configs/experimente_modelle.yaml`):

| Modell | Größe | Recall@5 Alltag | MRR Alltag | MRR Fach | AUC Fremdfragen | Fremdfragen über Schwelle |
|---|---|---|---|---|---|---|
| multilingual-e5-small | ca. 470 MB | 0,45 [0,30; 0,60] | 0,32 | 0,86 | 0,98 | 1 von 15 |
| multilingual-e5-base | ca. 1,1 GB | 0,63 [0,48; 0,78] | 0,47 | 0,87 | 0,97 | 5 von 15 |
| **multilingual-e5-large** | ca. 2,2 GB | **0,78** [0,65; 0,90] | **0,57** | **0,93** | 0,97 | 5 von 15 |

Gegenüber e5-small verbessert e5-large Recall@5 bei Alltagsfragen um 0,33 [0,15; 0,50]. Beim
Aussortieren fachfremder Fragen ist es an der Schwelle etwas schlechter. Die AUC-Intervalle
überlappen sich jedoch stark. Das Modell wurde auf denselben 40 Items ausgewählt, an denen es
gemessen wird. Bei drei Kandidaten und diesen Abständen ist das vertretbar, die Werte für e5-large
sind dadurch aber eher etwas zu optimistisch.

### Getestet und verworfen: Umformulierung der Anfrage

Ein lokales Sprachmodell (Ollama, `qwen2.5:3b` und `qwen2.5:7b`) sollte Alltagsfragen vor der Suche
in die Fachsprache des Gesetzes übersetzen. Bei der Belehrungsfrage behielten beide Modelle
„Strahlenschutzbelehrung“ bei, statt „Unterweisung“ zu verwenden. Das 3B-Modell veränderte zudem
die Bedeutung („Welche Anzahl …“ statt „Wie oft …“). Das 7B-Modell erfand eine Fundstelle
(„§ 7 Absatz 1 StrahlSchG“). Die Suche wurde dadurch nicht besser. Der Ansatz wurde nur an
Einzelbeispielen geprüft, nicht auf dem ganzen Fragenset, und ist nicht mehr im Code.

## Aufbau

```
src/strlrag/
  korpus.py      XML von gesetze-im-internet.de -> Vorschriften (Titel, Gliederung, Absätze)
  text.py        Normalisierung, Stoppwörter, CISTEM-Stemmer, Satzzerlegung
  chunking.py    ein Absatz je Chunk, Titel der Vorschrift als Präfix
  retrieval.py   Embeddings (Haupt-Retriever), BM25 (Basislinie)
  antwort.py     Prompt mit nummerierten Quellen, Backends (extraktiv, OpenAI-kompatibel, Mock),
                 Enthaltung, Prüfung der Zitate
  evaluation.py  Rang auf Paragraphenebene, Recall@k, MRR, Cluster-Bootstrap, AUC
  daten.py       Pfade und Laden des Standardkorpus
scripts/
  lade_korpus.py     lädt die XML-Dateien nach data/roh/
  pruefe_fragen.py   prüft Soll-Vorschriften und Stichworte gegen den Wortlaut
  experimente.py     wertet die Retriever einer Konfiguration aus
  frage.py           Kommandozeile: Frage stellen, Antwort mit Quellen
configs/
  experimente.yaml          Embeddings gegen BM25 (Ergebnisse oben)
  experimente_modelle.yaml  Vergleich der Embedding-Modelle
  experimente_llm.yaml      Antworten eines Sprachmodells gegen die extraktive Antwort
data/eval/fragen.yaml        40 Items (fach + alltag), 15 Fremdfragen
data/eval/pruefung_llm.yaml  manuelle Prüfung der nicht korrekt gewerteten Antworten
tests/               51 Tests auf einem synthetischen Testgesetz
```

## Schnellstart

Auf Linux ohne GPU zuerst die kleinere CPU-Version von PyTorch installieren, sonst lädt pip die
deutlich größere GPU-Version:

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev,dense]"
pytest -q                                   # läuft ohne Rohdaten und ohne Modell
python scripts/lade_korpus.py               # StrlSchG + StrlSchV nach data/roh/
python scripts/pruefe_fragen.py             # Fragenset gegen den aktuellen Wortlaut prüfen
python scripts/experimente.py               # lädt beim ersten Lauf das Modell (ca. 2,2 GB)
python scripts/frage.py "Wie oft ist die Unterweisung zu wiederholen?"
```

Die Embeddings des Korpus werden je Modell in `data/cache/` zwischengespeichert. Weitere Läufe
kodieren nur noch die Fragen. Ein anderes Modell lässt sich über den Namen wählen, z. B.
`--retriever dense:intfloat/multilingual-e5-small`.

Ausgabe mit der Basislinie (`--retriever bm25`) auf eine Fachfrage:

```
Danach ist die Unterweisung mindestens einmal im Jahr zu wiederholen. [1] Die Unterweisung hat mündlich zu erfolgen. [2]

[1] StrlSchV § 63 Abs. 1 (Unterweisung)
[2] StrlSchV § 63 Abs. 3 (Unterweisung)
...
```

Und auf eine Alltagsfrage, mit einer selbstsicheren, aber falschen Antwort:

```
$ python scripts/frage.py --retriever bm25 "Wie viel Strahlung darf ich als Beschäftigter pro Jahr höchstens abbekommen?"
Die Genehmigung wird auf längstens fünf Jahre befristet. [2] ...

[1] StrlSchG § 25 Abs. 1 (Genehmigungsbedürftige Beschäftigung in fremden Anlagen oder Einrichtungen)
...
```

Die Zitate sind formal korrekt, die Quellen passen aber nicht zur Frage. Mit Embeddings steht für
diese Frage § 78 StrlSchG auf Platz 1.

Antworten mit einem lokalen Sprachmodell, z. B. über [Ollama](https://ollama.com):

```bash
export LLM_BASE_URL=http://localhost:11434/v1 LLM_MODEL=qwen2.5:7b
python scripts/frage.py --backend llm "Wie oft muss ich zur Strahlenschutz-Belehrung?"
python scripts/experimente.py --config configs/experimente_llm.yaml \
       --aus ergebnisse_llm --abbildungen docs/abbildungen_llm
```

Der zweite Befehl stellt alle 80 Korpus- und 15 Fremdfragen an das Modell. Die Antworten werden in
`data/cache/llm/` gespeichert, ein abgebrochener Lauf setzt dort wieder an. Im Prompt wird jede
Quelle auf 250 Wörter gekürzt, damit er in das Kontextfenster kleiner lokaler Modelle passt.

## Methode

**Korpus.** XML-Fassungen von [gesetze-im-internet.de](https://www.gesetze-im-internet.de)
(StrlSchG Stand 02.11.2026, StrlSchV Stand 05.08.2025): 459 Vorschriften einschließlich
28 Anlagen, rund 118.000 Wörter. Inhaltsübersichten, Fußnoten und weggefallene Paragraphen
werden übersprungen. Die Texte sind amtliche Werke (§ 5 UrhG). Sie liegen trotzdem nicht im
Repository, damit stets die aktuelle Fassung geladen wird.

**Zerlegung.** Ein Absatz je Chunk, insgesamt 1.397. Absätze sind die Einheit, auf die
Rechtstexte selbst verweisen („§ 78 Abs. 4“), und erlauben präzise Quellenangaben. Der Titel der
Vorschrift wird als Präfix mitindexiert, weil viele Absätze für sich nicht sagen, worum es geht.

**Retriever.**
- **Embeddings:** `intfloat/multilingual-e5-large` über sentence-transformers, mit den von E5
  erwarteten Präfixen „query: “ und „passage: “, normiert und per Kosinus-Ähnlichkeit verglichen.
- **BM25** (k1 = 1,2, b = 0,75) als Basislinie: Tokens mit gefalteten Umlauten, entfernten
  Stoppwörtern und CISTEM-Stemming (Weissweiler & Fraser 2017).

**Bewertung.** Ein Treffer zählt auf Paragraphenebene: Liegt ein abgerufener Absatz in einer
Soll-Vorschrift, ist die Frage beantwortet. Mehrere Absätze derselben Vorschrift zählen für den
Rang nur einmal. Unsicherheiten stammen aus einem Cluster-Bootstrap über Items mit 2.000
Ziehungen, bei dem beide Formulierungen eines Items gemeinsam gezogen werden. Vergleiche laufen
gepaart auf denselben Fragen.

**Antwort.** Die extraktive Basislinie wählt die zwei Sätze aus den fünf besten Absätzen, die
der Frage per TF-IDF am ähnlichsten sind, und hängt die Quellennummer an. Für ein Sprachmodell
wird ein Prompt mit nummerierten Quellen und einer festen Enthaltungsformel gebaut.
`Antwort.ungueltige_zitate` meldet Quellennummern, die es im Kontext nicht gibt.

**Fragenset.** Zu jedem Item gibt es Soll-Vorschriften, ein Stichwort aus dem Wortlaut, z. B.
„300 Becquerel je Kubikmeter“, und kurze Antwortmuster wie „300 Becquerel“ oder „300 Bq“. Eine
Antwort gilt als korrekt, wenn das System sich nicht enthält und eines der Muster im Antworttext
steht. Das ist ein grobes Kriterium: Es erkennt keine falschen Zusatzaussagen und übersieht
korrekte, aber anders formulierte Antworten. `scripts/pruefe_fragen.py` stellt sicher, dass jedes
Stichwort und mindestens ein Antwortmuster je Item tatsächlich in einer Soll-Vorschrift steht. Der
zeitgesteuerte CI-Job wiederholt diese Prüfung wöchentlich gegen die aktuelle Gesetzesfassung.

![Enthaltung](docs/abbildungen/enthaltung.png)

## Grenzen und nicht Getestetes

- **Das Fragenset ist klein und verzerrt.** 40 Items von einer Person. Die Fachfragen entstanden
  nach Lektüre der Vorschrift und sind dadurch für wortbasierte Verfahren eher zu leicht. Die
  Alltagsfragen vermeiden die Begriffe der Vorschrift bewusst und sind dadurch eher zu schwer.
  Echte Nutzerfragen dürften dazwischen liegen. Die Intervalle sind entsprechend breit.
- **Modell und Schwelle sind nicht unabhängig gewählt.** Beides wurde auf denselben Fragen
  bestimmt, an denen es gemessen wird. Für den Einsatz bräuchte es ein eigenes Fragenset dafür.
- **Bewertet wird nur, ob die richtige Vorschrift gefunden wird.** Ob eine Antwort juristisch
  vollständig ist, misst die Auswertung nicht. Verweise zwischen Vorschriften, etwa „nach § 121
  Absatz 1“, werden nicht aufgelöst.
- **Lange Absätze werden abgeschnitten.** E5 verarbeitet höchstens 512 Tokens. Etwa 36 der 1.397
  Absätze haben mehr als 300 Wörter und dürften gekürzt in die Embeddings eingehen, vor allem
  Tabellen in den Anlagen.
- **Getestet:** Embeddings und Sprachmodell (Ollama) wurden lokal unter Python 3.12 ausgeführt,
  in den Tests nur mit Ersatzmodellen. Die Entwicklungsumgebung lief unter Python 3.13.
- **Ein Sprachmodell, ein Lauf.** Gemessen wurde nur qwen2.5:7b mit Temperatur 0. Andere oder
  größere Modelle können sich deutlich anders verhalten. Alle Antworten stehen im Wortlaut in
  `ergebnisse_llm/antworten.csv`.
- **Die automatische Antwortbewertung ist grob.** Sie übersieht korrekte, anders formulierte
  Antworten (9 Fälle) und erkennt keine falschen Zusatzaussagen in sonst korrekt gewerteten
  Antworten. Diese wurden nicht von Hand geprüft.
- **Nicht ausgeführt:** `scripts/lade_korpus.py` (die Rohdaten wurden manuell heruntergeladen)
  und die CI.

## Nächste Schritte

1. Sicherheitsrelevante Themen gezielt prüfen: Für Schutzpflichten, etwa bei Schwangerschaft,
   Stillzeit oder Jugendlichen, eigene Testfragen anlegen und Antworten dazu kennzeichnen oder
   nur mit Hinweis auf fachliche Prüfung ausgeben (siehe q36).
2. Weitere oder größere Sprachmodelle gegen qwen2.5:7b messen, besonders bei den 4
   übervorsichtigen Enthaltungen und den 3 falschen Antworten.
3. Das Fragenset von weiteren Personen formulieren lassen, Fragen ohne Lektüre der Vorschriften
   ergänzen und einen Teil für die Wahl von Modell und Schwelle zurückhalten.
4. Die extraktive Basislinie stärken: Bisher vergleicht sie nur die Sätze selbst mit der Frage,
   ohne den Titel der Vorschrift. Bei q02 („Berufslebensdosis“) steht der Begriff nur im Titel von
   § 77 StrlSchG, nicht im Satz mit „400 Millisievert“. Deshalb wählt die extraktive Antwort dort
   falsche Sätze, obwohl § 77 auf Platz 1 abgerufen wird. Der Titel sollte beim Vergleich mitzählen.

## Lizenz

Der Code steht unter der Lizenz in [`LICENSE`](LICENSE). Die Gesetzestexte sind nicht im
Repository enthalten. `scripts/lade_korpus.py` lädt sie von gesetze-im-internet.de; als amtliche
Werke (§ 5 UrhG) sind sie urheberrechtlich frei.
