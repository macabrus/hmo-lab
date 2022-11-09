import sys
import csv
import collections

instance = open(sys.argv[1],"r")
solution = open(sys.argv[2],"r")

#read instance file
reader = csv.reader(instance)
data = list(reader)

#read solution file
team = solution.readline().rstrip("\n").split(",")
subs = solution.readline().rstrip("\n").split(",")

#first team information
score = 0
price = 0
clubs = []
positions_team = []
for pt in team:
    for pi in data:
        if(pi[0]==pt):
            score = score + int(pi[4])
            price = price + float(pi[5])
            clubs.append(pi[3])
            positions_team.append(pi[1])

#subs information
positions_subs = []
for ps in subs:
    for pi in data:
        if(pi[0]==ps):
            price = price + float(pi[5])
            clubs.append(pi[3])
            positions_subs.append(pi[1])
            
#validate information
if(len(team)!=11):
    print("Wrong number of players in the first team")
if(len(subs)!=4):
    print("Wrong number of players on the bench")
if(price > 100):
    print("Over the budget")
for i in collections.Counter(clubs).most_common():
    if(int(i[1])>3):
        print("To many players from", i[0])
for i in collections.Counter(positions_team).most_common():
    if(i[0]=='GK' or 'GK' not in positions_team):
        if(int(i[1])!=1):
           print("Wrong number of goalkeepers in the first team")
    if(i[0]=='DEF' or 'DEF' not in positions_team):
        if(int(i[1])<3 or int(i[1])>5):
           print("Wrong number of defenders in the first team")
    if(i[0]=='MID' or 'MID' not in positions_team):
        if(int(i[1])<2 or int(i[1])>5):
            print("Wrong number of midfielders in the first team")
    if(i[0]=='FW' or 'FW' not in positions_team):
        if(int(i[1])<1 or int(i[1])>3):
            print("Wrong number of forwards in the first team")
for i in collections.Counter(positions_subs).most_common():
    if(i[0]=='GK' or 'GK' not in positions_subs):
        if(int(i[1])!=1):
            print("Wrong number of goalkeepers on the bench")
positions = positions_team + positions_subs
for i in collections.Counter(positions).most_common():
    if(i[0]=='GK' or 'GK' not in positions):
        if(int(i[1])!=2):
           print("Wrong number of goalkeepers in the 15-player squad")
    if(i[0]=='DEF' or 'DEF' not in positions):
        if(int(i[1])!=5):
           print("Wrong number of defenders in the 15-player squad")
    if(i[0]=='MID' or 'DEF' not in positions):
        if(int(i[1])!=5):
            print("Wrong number of midfielders in the 15-player squad")
    if(i[0]=='FW' or 'FW' not in positions):
        if(int(i[1])!=3):
            print("Wrong number of forwards in the 15-player squad")
print("Price:",price)
print("Score:",score)

instance.close()
solution.close()
