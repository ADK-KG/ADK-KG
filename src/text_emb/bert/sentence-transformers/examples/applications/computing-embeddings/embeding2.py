from sentence_transformers import SentenceTransformer
#model = SentenceTransformer('paraphrase-distilroberta-base-v1')
model = SentenceTransformer('paraphrase-xlm-r-multilingual-v1')
content = []
for k in range(1,8):
    files = str(k) + ".txt"
    f1 = open(files)             
    line = f1.readline()
    while line: 
        line = line.strip('\n')
        item = str(line)
        content.append(item)
        line = f1.readline()
    print (len(content))

files = "0.txt"
f1 = open(files)             
line = f1.readline()
while line: 
    line = line.strip('\n')
    item = str(line)
    content.append(item)
    line = f1.readline()
print (len(content))
sentence_embeddings = model.encode(content)

with open("data__.txt", 'w') as fw:
    n = 0
    for sentence, embedding in zip(content, sentence_embeddings):
        for k in embedding:
            fw.write ('%s,'%(k))
        fw.write ('\n')
        n += 1