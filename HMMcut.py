

def viterbi(text, states,start_p, trans_p, emit_p):
    V = [{}]
    path = {}

    for y in states:   #INITIALIZE
        
        V[0][y] = start_p[y] * emit_p[y].get(text[0],0)  

        path[y] = [y]
    for t in range(1,len(text)):
        V.append({})
        newpath = {}
        
        newChar = text[t] not in emit_p['S'].keys() and text[t] not in emit_p['M'].keys() and text[t] not in emit_p['E'].keys() and text[t] not in emit_p['B'].keys()

        for y in states:      #FOR EVERY STATE
            
            emitP = emit_p[y].get(text[t], 0) if not newChar else 1.0
            (prob, state) = max([(V[t-1][y0] * trans_p[y0].get(y,0) * emitP ,y0) for y0 in states if V[t-1][y0]>0])
            V[t][y] =prob
            newpath[y] = path[state] + [y]
        path = newpath  #STORE PATH
    (prob, state) = max([(V[len(text) - 1][y], y) for y in states])  #MAX PROBABILITY FOR Y IN THE END

    return (prob, path[state])  #RETURN PROBABILITY AND PATH
        
        
        
def load_model(f_name):
    fp=open(f_name,'rb').read()
    return eval(fp)


state_list = ['B','M','E','S']
prob_start = load_model("prob_start.py")
prob_trans = load_model("prob_trans.py")
prob_emit = load_model("prob_emit.py")

        
def cut(text):

    prob, pos_list = viterbi(text,state_list,prob_start, prob_trans, prob_emit)
    
    print(prob,pos_list)


text = '这是一个非常棒的地方！'
cut(text)