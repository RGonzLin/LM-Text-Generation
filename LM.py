# Import libraries
import numpy as np
import tensorflow as tf
import re
import pickle
import math
from keras_nlp.metrics import Perplexity

def WordAnalyzer(text):

    # Split on blank spaces, punctuation, and numbers
    words = re.findall(r'\w+|[^\w\s]+', text)

    for i, word in enumerate(words):

        # Split numbers and letters (e.g. 1st --> 1 and st)
        if re.search(r'\d+\w+', word):
            match = re.search(r'(\d+)(\w+)', word)
            words[i] = match.group(1)
            words.insert(i+1, match.group(2))

        # Replace words containing uppercase letters with all-lowercase version
        if re.search(r'[A-Z]', word):
            words[i] = word.lower()

        # Split numbers into digits
        if word.isdigit():
            words[i:i+1] = list(word)

    # Return the list of words
    return words

class ToyLM_ngram():

    def GetSentences(self,path):

        # Extract sentences from file
        sentences = []

        # Open file and extract sentences
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                sentences.append(line)

        # Return sentences
        return sentences
    
    def ProcessData(self,texts,analyzer=WordAnalyzer):
        
        # Split each text into words using the specified analyzer
        wordLists = []
        for text in texts:
            words = analyzer(text)
            wordLists.append(words)
        
        # Return the word lists
        return wordLists
    
    def CreateModel(self,texts,n,analyzer=WordAnalyzer):
        
        # let n be available for other methods
        self.n = n

        # Split each text into words using the specified analyzer
        wordLists = self.ProcessData(texts,analyzer)

        # Create a dictionary to store the n-grams and their counts
        self.nGramCounts = {}

        # Loop over the tokens and create n-grams 
        for tokens in wordLists:
            for i in range(len(tokens) - n + 1):
                nGram = ' '.join(tokens[i:i+n])

                # Add the n-gram to the dictionary and increment its count
                if nGram in self.nGramCounts:
                    self.nGramCounts[nGram] += 1
                else:
                    self.nGramCounts[nGram] = 1

        # Count each n-gram
        for nGram, count in self.nGramCounts.items():
            self.nGramCounts[nGram] = count

    def Test(self,texts,analyzer=WordAnalyzer):

        # Create a Perplexity object
        perplexity = Perplexity(name="perplexity")

        # Split each text into words using the specified analyzer
        wordLists = self.ProcessData(texts,analyzer)

        # Initialize counters
        totalTests = 0
        validTests = 0

        for sentence in wordLists:

            # Create all posible windows of length n (except the last one)
            windows = [sentence[i:i+self.n] for i in range(len(sentence) - self.n + 1)][:-1]
            # Pop the last element of each window and save it as a label
            labels = [window.pop() for window in windows]
            # Create a string for all windows where each element is separated by a space
            windows = [' '.join(window) for window in windows]

            # Loop over all windows and labels
            for window, label in zip(windows,labels):

                # Predict the next word
                containsMgram = self.Predict(window,returnContainsMgram=True)

                # If the prediction is invalid get the full n-gram counts
                if containsMgram == None:
                    containsMgram = self.nGramCounts

                # Create a dictionary with the n-grams that contain the label
                yTrueDict = {key: value for key, 
                         value in containsMgram.items() if key.endswith(label)}
                
                # Increase the total number of tests counter 
                totalTests += 1
                
                # Check if the label is contained in the n-grams (for safety)
                if len(yTrueDict) == 1:

                    mapping = {}
                    count = 0
                    # Create a mapping from the n-grams to the corresponding index
                    for key in containsMgram.keys():
                        if key.endswith(label):
                            mapping[key] = count
                            mapping[label] = count
                        else:
                            mapping[key] = count
                        count += 1

                    # Create a tensor with the predictions and the true label
                    # Note the perplexity has to be calculated in the loop because
                    # not every yPred is of the same size, which would lead 
                    # to a non-recalngular array
                    yPred = [containsMgram[key] for key in containsMgram.keys()]
                    yPred = tf.constant([[yPred]], dtype=tf.float32)
                    yTrue = mapping[label]
                    yTrue = tf.constant([[yTrue]], dtype=tf.int32)

                    # Update the perplexity
                    perplexity.update_state(yTrue,yPred)
                    
                    # Increase the valid tests counter
                    validTests += 1

        # Print the average perplexity
        print('The average perplexity is: ', perplexity.result().numpy())
        # Print the percentage of valid tests
        print('* Note: The percentage of valid tests is: ', validTests/totalTests*100, '%')
                        
    def Predict(self,inputPredict,maxLength=30,numberOfConsideredWords=None,analyzer=WordAnalyzer,
                returnContainsMgram=False):

        # Split the input text into tokens using the specified analyzer
        tokens = analyzer(inputPredict)

        # Get the last n-1 (m) tokens as a string
        mGram = ' '.join(tokens[-(self.n-1):])
        # Get the head in order to attach it to the begining of the final prediction
        head = ' '.join(tokens[:-(self.n-1)])
    
        # Initialize the predictions
        predictions = mGram

        # Pre-compute the probabilities of all nGrams
        nGramCounts = self.nGramCounts
        totalNgrams = sum(nGramCounts.values())
        for nGram, count in nGramCounts.items():
            nGramCounts[nGram] = count / totalNgrams
        preChoices = list(nGramCounts.keys())
        preProbabilities = list(nGramCounts.values())

        nextToken = ''
        # Generate text until maxLength is reached
        while len(predictions.split()) < maxLength and nextToken != '.': 

            # Find all n-grams starting with the m-gram
            containsMgram = {key: value for key, value in self.nGramCounts.items() if key.startswith(mGram)}

            # If the m-gram is contained in any n-gram, compute the probabilities and make a choice 
            if len(containsMgram) > 0:
                totalMgrams = sum(containsMgram.values())
                for mGram, count in containsMgram.items():
                    containsMgram[mGram] = count / totalMgrams
                choices = list(containsMgram.keys())
                probabilities = list(containsMgram.values())
                # If the number of considered words is not specified, chose amongst all options
                if numberOfConsideredWords == None:
                    nextMgram = np.random.choice(choices, p=probabilities)
                # Else set a cutoff and set the probabilities below the cutoff to 0
                else:
                    # If the number of n-grams is greater than the number of considered words,
                    # set a cutoff to only consider the n-grams with the highest probabilities
                    if len(probabilities) > numberOfConsideredWords:
                        cutoff = np.partition(probabilities,-numberOfConsideredWords)[len(probabilities)
                                                                            -numberOfConsideredWords]
                        probabilities = [0 if x < cutoff else x for x in probabilities]
                        probabilities = probabilities/np.sum(probabilities)
                        nextMgram = np.random.choice(choices, p=probabilities)
                    # Else consider all n-grams
                    else:
                        nextMgram = np.random.choice(choices, p=probabilities)
                
                # Return n-grams containing the m-gram if specified
                if returnContainsMgram == True:
                    return containsMgram
                
            # Else compute the probabilities of all n-grams and make a choice
            else:
                # If the number of considered words is not specified, chose amongst all options
                if numberOfConsideredWords == None:
                    nextMgram = np.random.choice(preChoices, p=preProbabilities)
                # Else set a cutoff and set the probabilities below the cutoff to 0
                else:
                    cutoff = np.partition(preProbabilities,-numberOfConsideredWords)[len(preProbabilities)
                                                                          -numberOfConsideredWords]
                    probabilities = [0 if x < cutoff else x for x in preProbabilities]
                    probabilities = probabilities/np.sum(probabilities)
                    nextMgram = np.random.choice(preChoices, p=probabilities)

                # Return None instead n-grams containing the m-gram if specified,
                # as returing such n-grams would ultimately lead to an infinite perpelexity value
                if returnContainsMgram == True:
                    return None
        
            # Append the next token to the predictions and update the m-gram
            nextToken = nextMgram.split()[-1]
            predictions += ' ' + nextToken
            mGram = ' '.join(predictions.split()[-(self.n-1):]) + ' '

        # Append the input text to the begining of the predictions
        predictions = head + ' ' + predictions
        
        # Return the predictions
        return predictions
    
    def Save(self,path):

        # Save the n-gram counts and n in the specified path
        with open(path, 'wb') as handle:
            pickle.dump((self.nGramCounts,self.n),
                        handle,protocol=pickle.HIGHEST_PROTOCOL)

    def Load(self,path):
        
        # Load the n-gram counts and from the specified path
        with open(path, 'rb') as handle:
            loader = pickle.load(handle)
        
        # Set the n-gram counts and n
        self.nGramCounts = loader[0]
        self.n = loader[1]

class ToyLM_LSTM():

    def GetSentences(self,path):

        # Extract sentences from file
        sentences = []

        # Open file and extract sentences
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                sentences.append(line)

        # Return sentences
        return sentences
    
    def ProcessData(self,sentences,trainSet=False,windowSize=4,analyzer=WordAnalyzer,numWords=None,
                    tokenizerPath = 'tokenizer.pickle'):

        # Let the size of the window be availible to other methods
        self.windowSize = windowSize

        # Create tokenizer only for the training set
        if trainSet==True:
            
            # Create tokenizer object
            self.tokenizer = tf.keras.preprocessing.text.Tokenizer(oov_token='OOV',
                                                                   analyzer=analyzer,num_words=numWords)

            # Build vocabularies for set
            self.tokenizer.fit_on_texts(sentences)

            # Save tokenizer and window size
            with open(tokenizerPath, 'wb') as handle:
                pickle.dump((self.tokenizer,self.windowSize),handle,protocol=pickle.HIGHEST_PROTOCOL)

        # Encode inputs for set
        sentencesEncoded = self.tokenizer.texts_to_sequences(sentences) 

        # Pad inputs
        sentencesPadded = []
        for sentence in sentencesEncoded:
            sentencePadded = np.concatenate((np.zeros(windowSize), np.array(sentence)))
            sentencesPadded.append(sentencePadded)

        # Create time windows and labels
        window = []
        labels = []
        for sentence in sentencesPadded:
            for i in range(0,len(sentence)-windowSize):
                window.append(sentence[i:i+windowSize])
                labels.append(sentence[i+windowSize])
                
        # Convert to numpy arrays
        window = np.array(window,dtype=np.int16)
        labels = np.array(labels,dtype=np.int16) 
        
        # Return inputs and labels
        return window, labels

    def CreateModel(self,embeddingSize=None,lstmUnits=64,dropout=0.2,recurrentDropout=0.2,
                    printSummary=False,vsize=None):

        # Get size of vocabulary if none is specified 
        # Make vsize to be available for other methods
        if vsize == None:
            self.vsize = len(self.tokenizer.word_index)+1
        else: 
            self.vsize = vsize #+ 1
        
        # Define embedding size
        if embeddingSize == None:
            embeddingSize = int(1.6*math.sqrt(self.vsize)) # Heuristic for embedding size
        else:
            embeddingSize = embeddingSize

        # Define inputs
        inputs = tf.keras.Input(shape=(self.windowSize,), dtype=np.int16)                               

        # Create network 
        x = tf.keras.layers.Embedding(self.vsize,embeddingSize,mask_zero=True)(inputs) 
        # Create LSTM layer
        x = tf.keras.layers.LSTM(lstmUnits,return_sequences=False,dropout=dropout,
                                recurrent_dropout=recurrentDropout)(x) 
        # Create desnse layer
        outputs = tf.keras.layers.Dense(self.vsize,activation='softmax')(x) # As many neurons 
                                                                            # as words in vocabulary 

        # Create model
        self.model = tf.keras.Model(inputs=inputs,outputs=outputs)

        # Print model summary if requested
        if printSummary==True:
            print(self.model.summary())
    
    def Train(self,inputTrain,targetTrain,valTuple=None,epochs=10,
              optimizer=tf.keras.optimizers.Adam(),metrics=Perplexity(name="perplexity"),batchSize=None,
              saveBest=True,patience=5,modelName='model.h5'):

        # Compile model
        self.model.compile(optimizer=optimizer,loss='sparse_categorical_crossentropy',
                           metrics=metrics) 
        
        # Define callbacks 
        earlyStopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=patience)
        modelCheckpoint = tf.keras.callbacks.ModelCheckpoint(filepath=modelName,saveBest=saveBest,
                                                             monitor="val_loss")
        callbacks = [earlyStopping, modelCheckpoint]

        # Train model
        self.model.fit(inputTrain,targetTrain,validation_data=valTuple,
                       epochs=epochs,batch_size=batchSize,callbacks=callbacks)
    
    def Test(self,inputTest,targetTest):

        # Test model 
        results = self.model.evaluate(inputTest,targetTest,verbose=0)

        # Print perplexity
        print('The average perplexity is: ', results[1])
    
    def Predict(self,inputPredict,maxLength=30,numberOfConsideredWords=None):

        # If the number of considered words is set to None, consider all words
        if numberOfConsideredWords == None:
            numberOfConsideredWords = self.vsize - 1

        # If the input is empty, create an empty context window
        if str.isspace(inputPredict) == True or inputPredict == '': 
            # Create an empty context window
            lastWindow = np.zeros(self.windowSize,dtype=np.int16)
            lastWindow = np.array([lastWindow.tolist()],dtype=np.int16)
        else:
            # Process input
            windows, labels = self.ProcessData([inputPredict],windowSize=self.windowSize,analyzer=WordAnalyzer)
            # Get last window
            lastWindow = np.array([windows[-1]],dtype=np.int16)

        # Create list with all posible tokens
        tokens = list(self.tokenizer.word_index.keys())[:self.vsize-1] 

        # Get tokens form windows
        predictions = []

        c = 0
        prediction = ''
        while c < maxLength and prediction != '.':
            c += 1
            # Get probabilities
            probabilities = self.model.predict(lastWindow,verbose=0)[:, 1:] 
            # Reduce dimension by 1
            probabilities = probabilities[0]
            # Set up a cutoff value and take into account only the most probable words
            cutoff = np.partition(probabilities,-numberOfConsideredWords)[len(probabilities)
                                                                          -numberOfConsideredWords]
            probabilities[probabilities<cutoff] = 0
            # Normalize probabilities
            probabilities = probabilities/np.sum(probabilities)
            # Choose next word
            prediction = np.random.choice(tokens,1,p=probabilities)
            predictions.append(prediction)
            
            # Update window 
            tokenizedPred = int(self.tokenizer.texts_to_sequences(prediction)[0][0])
            lastWindow = lastWindow.tolist()
            lastWindow = lastWindow[0][1:]
            lastWindow.append(tokenizedPred)
            lastWindow = np.array([lastWindow])

        # Convert each numpy array to a string
        predictions = [word.item() for word in predictions]

        # Construct predicted sentence
        if str.isspace(inputPredict) == True or inputPredict == '':
            predictedSentence = ' '.join(predictions)
        else:
            predictedSentence = inputPredict + ' ' + ' '.join(predictions)

        # Return predicted sentence
        return predictedSentence

    def Save(self,path):
        
        # Save model in the specified path
        self.model.save(path)

    def Load(self,modelPath,tokenizerPath):

        # Load model from the specified path
        self.model = tf.keras.models.load_model(modelPath)

        # Load tokenizer and window size from the specified path
        with open(tokenizerPath, 'rb') as handle:
            loader = pickle.load(handle)

        # Set tokenizer and window size
        self.tokenizer = loader[0]
        self.windowSize = loader[1]

        # Load vsize
        self.vsize = len(self.tokenizer.word_index)+1