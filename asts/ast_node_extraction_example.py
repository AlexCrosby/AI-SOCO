from asts.ast_vectorizer import ASTVectorizer
# Example node n-gram extractor
tree = '1 1{1 2{1 3{2 4{3 5{}}4 6{}5 7{6 8{3 5{}6 6{}}7 8{7 6{}3 5{}}7 8{7 6{}3 5{}}8 8{9 5{}8 6{}}10 8{3 5{}10 6{}}11 9{12 6{}13 10{14 5{}13 6{}}}15 9{16 6{}17 10{17 6{}14 5{}}}18 9{19 6{}17 10{14 5{}17 6{}}}20 11{21 10{14 5{}21 6{}}10 10{10 6{}3 5{}}}22 12{23 13{7 10{3 5{}7 6{}}24 14{}}25 15{7 10{7 6{}3 5{}}10 10{3 5{}10 6{}}}26 16{7 10{3 5{}7 6{}}}5 7{27 11{21 10{21 6{}14 5{}}28 11{29 17{30 18{8 10{9 5{}8 6{}}7 10{3 5{}7 6{}}}31 10{14 5{}31 6{}}}32 17{30 18{8 10{8 6{}9 5{}}7 10{7 6{}3 5{}}}33 10{33 6{}14 5{}}}}}}}34 9{35 6{}8 10{9 5{}8 6{}}36 19{8 10{9 5{}8 6{}}10 10{3 5{}10 6{}}}}37 13{6 10{3 5{}6 6{}}24 14{}}38 12{23 13{7 10{3 5{}7 6{}}24 14{}}25 15{7 10{7 6{}3 5{}}10 10{3 5{}10 6{}}}26 16{7 10{3 5{}7 6{}}}5 7{39 20{40 15{32 17{30 18{8 10{9 5{}8 6{}}7 10{7 6{}3 5{}}}33 10{33 6{}14 5{}}}29 17{30 18{8 10{8 6{}9 5{}}7 10{3 5{}7 6{}}}31 10{14 5{}31 6{}}}}41 9{42 6{}32 17{30 18{8 10{9 5{}8 6{}}7 10{7 6{}3 5{}}}33 10{14 5{}33 6{}}}29 17{30 18{8 10{9 5{}8 6{}}7 10{3 5{}7 6{}}}31 10{31 6{}14 5{}}}}}43 20{44 21{29 17{30 18{8 10{9 5{}8 6{}}7 10{3 5{}7 6{}}}31 10{14 5{}31 6{}}}6 10{3 5{}6 6{}}}45 13{6 10{3 5{}6 6{}}29 17{30 18{8 10{8 6{}9 5{}}7 10{7 6{}3 5{}}}31 10{14 5{}31 6{}}}}46 22{47 13{6 10{6 6{}3 5{}}32 17{30 18{8 10{8 6{}9 5{}}7 10{7 6{}3 5{}}}33 10{33 6{}14 5{}}}}}}}}48 23{49 10{14 5{}49 6{}}6 10{6 6{}3 5{}}}}}}}'

vec = ASTVectorizer.get_ngrams(tree, 2, False) # Extract full bi-grams counts.
print(vec)