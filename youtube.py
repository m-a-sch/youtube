"""
Usage: youtube.py <youtube_id> [<question>]
  prints the script of the video when no question is provided
  prints youtube links into the video where the question is answered
  idea inspired by Yuvi https://github.com/yvrjsharma
"""
from youtube_transcript_api import YouTubeTranscriptApi
import sys
try:
  available = YouTubeTranscriptApi.list_transcripts(sys.argv[1])

  transcript = YouTubeTranscriptApi.get_transcript(sys.argv[1],languages=['en','de','fr',]) # this will only work for English
  transcriptAsString = ' '.join([i['text'] for i in transcript]).replace('[Music]',' ')
  transcriptAsWords = transcriptAsString.split(' ')
  tAWSize=len(transcriptAsWords)
except:
  print("No transcipt found")
  quit()


try:
  question=sys.argv[2]
  questionAsWords=question.split(' ')
  qAWSize=len(questionAsWords)
except:
  print("As no question was asked, we are done,\n",transcriptAsString,"\n bye")
  quit()

print("Now loading NLP, be patient")
from transformers import AutoTokenizer
from transformers import pipeline
from transformers import AutoModelForQuestionAnswering
import pandas as pd

#you can try different models and checkpoints
#checkPointQA = "deepset/minilm-uncased-squad2"
checkPointQA = "deepset/roberta-base-squad2"
#checkPointQA = "distilbert-base-cased-distilled-squad"

print("NLP loaded")


contexts=[]
i=0
while i < tAWSize:
  contexts.append(' '.join(transcriptAsWords[i:i+(5*qAWSize)]))
  i = i+qAWSize * 4


pipe = pipeline("question-answering",
                model=AutoModelForQuestionAnswering.from_pretrained(checkPointQA),
                tokenizer=AutoTokenizer.from_pretrained(checkPointQA))

answers=[]
for c in contexts:
  answers.append(pipe(question=question, context=c))
answers=pd.DataFrame(answers)
goodAnswers=answers.sort_values(by='score',ascending=False).head(2)['answer'].values

transcript=pd.DataFrame(transcript)

transcript['good']=None
def contains(a,b):
  for b1 in b:
    if b1 in a:
      return True
  return False

transcript['good']=transcript.apply(lambda x: contains(x['text'],goodAnswers),axis=1)


for s in transcript[transcript['good']]['start'].values:
  print(f"https://youtu.be/{sys.argv[1]}?t={int(s)}")
