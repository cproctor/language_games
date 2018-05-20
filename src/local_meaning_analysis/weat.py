# experimental setups from:

# caliskan, a., bryson, j. j. and narayanan, a. (2017) semantics
# derived automatically from language corpora contain human-like
# biases. science, 356 (6334). pp. 183-186. issn 0036-8075

experiments = {
    "weat1": [["pleasant1", "unpleasant1"], ["flowers", "insects"]],
    "weat2": [["pleasant1", "unpleasant1"], ["instruments", "weapons"]],
    "weat3": [["pleasant1", "unpleasant2"], ["euro_names3", "afro_names3"]],
    "weat4": [["pleasant1", "unpleasant2"], ["euro_names4", "afro_names4"]],
    "weat5": [["pleasant5", "unpleasant5"], ["euro_names4", "afro_names4"]],
    "weat6": [["career", "family"], ["male_names", "female_names"]],
    "weat7": [["male7", "female7"], ["math", "arts7"]],
    "weat8": [["male8", "female8"], ["science", "arts8"]],
    "weat9": [["temporary", "permanent"], ["mental_disease", "physical_disease"]],
    "weat10": [["pleasant5", "unpleasant5"], ["young_names", "old_names"]]
}

clusters = {
    # axis endpoints
    "pleasant1": ['caress', 'freedom', 'health', 'love', 'peace', 'cheer', 'friend', 'heaven', 
        'loyal', 'pleasure', 'diamond', 'gentle', 'honest', 'lucky', 'rainbow', 
        'diploma', 'gift', 'honor', 'miracle', 'sunrise', 'family', 'happy', 
        'laughter', 'paradise', 'vacation'
    ],
    "unpleasant1": ['abuse', 'crash', 'filth', 'murder', 'sickness', 'accident', 
        'death', 'grief', 'poison', 'stink', 'assault', 'disaster', 'hatred', 
        'pollute', 'tragedy', 'divorce', 'jail', 'poverty', 'ugly', 'cancer', 
        'kill', 'rotten', 'vomit', 'agony', 'prison'
    ],
    "unpleasant2": ['abuse', 'crash', 'filth', 'murder', 'sickness', 'accident', 
        'death', 'grief', 'poison', 'stink', 'assault', 'disaster', 'hatred', 
        'pollute', 'tragedy', 'bomb', 'divorce', 'jail', 'poverty', 'ugly', 
        'cancer', 'evil', 'kill', 'rotten', 'vomit'
    ],
    "pleasant5": ['joy', 'love', 'peace', 'wonderful', 'pleasure', 'friend', 
        'laughter', 'happy'
    ],
    "unpleasant5": ['agony', 'terrible', 'horrible', 'nasty', 'evil', 'war', 
        'awful', 'failure'
    ],
    "career": ['executive', 'management', 'professional', 'corporation', 'salary', 
        'office', 'business', 'career'
    ],
    "family": ['home', 'parents', 'children', 'family', 'cousins', 'marriage', 
        'wedding', 'relatives'
    ],
    "male7": ['male', 'man', 'boy', 'brother', 'he', 'him', 'his', 'son'],
    "female7": ['female', 'woman', 'girl', 'sister', 'she', 'her', 'hers', 'daughter'],
    "male8": ['brother', 'father', 'uncle', 'grandfather', 'son', 'he', 'his', 'him'],
    "female8": ['sister', 'mother', 'aunt', 'grandmother', 'daughter', 'she', 'hers', 'her'],
    "temporary": ['impermanent', 'unstable', 'variable', 'fleeting', 'short', 'brief', 
        'occasional'
    ],
    "permanent": ['stable', 'always', 'constant', 'persistent', 'chronic', 'prolonged', 
        'forever'
    ],

    # stimuli
    "flowers": ['aster', 'clover', 'hyacinth', 'marigold', 'poppy', 'azalea', 'crocus', 'iris', 'orchid', 'rose', 'daffodil', 'lilac', 'pansy', 'tulip', 'buttercup', 'daisy', 'lily', 'peony', 'violet', 'carnation', 'magnolia', 'petunia', 'zinnia'], 
    # bluebell, gladiola omitted
    "insects": ['ant', 'caterpillar', 'flea', 'locust', 'spider', 'bedbug', 'centipede', 'fly', 'maggot', 'tarantula', 'bee', 'cockroach', 'gnat', 'mosquito', 'termite', 'beetle', 'cricket', 'hornet', 'moth', 'wasp', 'dragonfly', 'horsefly', 'roach', 'weevil'], 
    # blackfly ommitted
    "instruments": ['bagpipe', 'cello', 'guitar', 'lute', 'trombone', 'banjo', 'clarinet', 'harmonica', 'mandolin', 'trumpet', 'bassoon', 'drum', 'harp', 'oboe', 'tuba', 'bell', 'fiddle', 'harpsichord', 'piano', 'viola', 'bongo', 'flute', 'horn', 'saxophone', 'violin'],
    "weapons": ['arrow', 'club', 'gun', 'missile', 'spear', 'axe', 'dagger', 'harpoon', 'pistol', 'sword', 'blade', 'dynamite', 'hatchet', 'rifle', 'tank', 'bomb', 'firearm', 'knife', 'shotgun', 'teargas', 'cannon', 'grenade', 'mace', 'slingshot', 'whip'],
    "euro_names3": ['adam', 'harry', 'josh', 'roger', 'alan', 'frank', 'justin', 'ryan', 'matthew', 'stephen', 'brad', 'greg', 'paul', 'jonathan', 'peter', 'amanda', 'courtney', 'heather', 'melanie', 'katie', 'betsy', 'kristin', 'nancy', 'stephanie', 'ellen', 'lauren', 'colleen', 'emily', 'megan', 'rachel'], 
    # Andrew, Jack omitted
    "afro_names3": ['alonzo', 'alphonse', 'jerome', 'leroy', 'torrance', 'darnell', 'lamar', 'lionel', 'deion', 'lamont', 'malik', 'terrence', 'tyrone', 'lavon', 'marcellus', 'wardell', 'nichelle', 'ebony', 'jasmine', 'tia', 'lakisha', 'latoya', 'yolanda', 'yvette'],
    # ['jamel' 'tyree' 'shereen' 'latisha' 'shaniqua' 'tanisha' 'malika'] omitted
    "euro_names4": ['brad', 'brendan', 'geoffrey', 'greg', 'brett', 'jay', 'matthew', 'neil', 'todd', 'allison', 'anne', 'carrie', 'emily', 'jill', 'laurie', 'kristen', 'meredith', 'sarah'], 
    "afro_names4": ['darnell', 'hakim', 'jermaine', 'kareem', 'jamal', 'leroy', 'rasheed', 'tyrone', 'aisha', 'ebony', 'kenya', 'lakisha', 'latoya'],
    # ['tremayne' 'keisha' 'latonya' 'tamika' 'tanisha'] omitted
   "euro_names5": ['brad', 'brendan', 'geoffrey', 'greg', 'brett', 'jay', 'matthew', 'neil', 'todd', 'allison', 'anne', 'carrie', 'emily', 'jill', 'laurie', 'kristen', 'meredith', 'sarah'],
    "afro_names5": ['darnell', 'hakim', 'jermaine', 'kareem', 'jamal', 'leroy', 'rasheed', 'tremayne', 'tyrone', 'aisha', 'ebony', 'keisha', 'kenya', 'latonya', 'lakisha', 'latoya', 'tamika', 'tanisha'],
    "male_names": ['john', 'paul', 'mike', 'kevin', 'steve', 'greg', 'jeff', 'bill'],
    "female_names": ['amy', 'joan', 'lisa', 'sarah', 'diana', 'kate', 'ann', 'donna'],
    "math": ['math', 'algebra', 'geometry', 'calculus', 'equations', 'computation', 'numbers', 'addition'],
    "arts7": ['poetry', 'art', 'dance', 'literature', 'novel', 'symphony', 'drama', 'sculpture'],
    "science": ['science', 'technology', 'physics', 'chemistry', 'einstein', 'nasa', 'experiment', 'astronomy'],
    "arts8": ['poetry', 'art', 'shakespeare', 'dance', 'literature', 'novel', 'symphony', 'drama'],
    "mental_disease": ['sad', 'hopeless', 'gloomy', 'tearful', 'miserable', 'depressed'],
    "physical_disease": ['sick', 'illness', 'influenza', 'disease', 'virus', 'cancer'],
    "young_names": ['tiffany', 'michelle', 'cindy', 'kristy', 'brad', 'eric', 'joey', 'billy'],
    "old_names": ['ethel', 'gertrude', 'agnes', 'cecil', 'mortimer', 'edgar']
    # bernice, wilbert removed
}
