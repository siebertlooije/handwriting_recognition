import pickle


with open('voca_shen.txt') as f:
    voca = []
    for line in f:
        #result = line

        result = ''.join([i for i in line if not i.isdigit()])
        result = result.replace(" ", "")
        #print result
        result = result.rstrip()
        voca.append(result)
    with open('shen_vocabulary.pickle', 'wb') as handle:
        pickle.dump(voca, handle)