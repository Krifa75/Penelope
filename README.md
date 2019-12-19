# Penelope

Penelope is an IA which will try to translate mangas (for now in english or french)

For now we only detect text using yolo but the result is still not satisfying.<br>
The reason is we trained the model with small datasets (~1000 images) 

![Alt text](output.jpg?raw=true "Texts predicted")

We will re-train yolo but with a bigger datasets, and next create an ocr with CRNN. 