from deep_translator import GoogleTranslator
import numpy as np
import pandas as pd

df = pd.read_csv('__.csv', header= None )
Translate= np.empty(len(df.index),dtype=object)
Translate= pd.DataFrame(Translate)

for i in df.index:
        x = str(df.iloc[i, 0])
        if (len(x) > 5000):
                x1 = x[0:4998]
                x2 = x[4999:len(x)]
                t_x1 = GoogleTranslator(source='it', target='el').translate(x1) #
                t_x2 = GoogleTranslator(source='it', target='el').translate(x2)
                t = t_x1 + t_x2
        else:
                Translate.iloc[i,0] = GoogleTranslator(source='it', target='el').translate(df.iloc[i,0])
                print(Translate.iloc[i,0])
Translate.to_csv('___.csv', index=False, header=False)

def MEGALITERES_PROTASEIS_20000():

    from deep_translator import GoogleTranslator
    import numpy as np
    import pandas as pd

    df = pd.read_csv('Spanish_NoHate_Negative.csv', header= None )
    Translate= np.empty(len(df.index),dtype=object)
    Translate= pd.DataFrame(Translate)

    for i in df.index:
        x = str(df.iloc[i, 0])
        if (len(x) < 5000):
            Translate.iloc[i, 0] = GoogleTranslator(source='es', target='el').translate(df.iloc[i, 0])
            print(Translate.iloc[i, 0])
        else:

            t_x3 = " "
            t_x4 = " "
            t_x5 = " "
            t_x6 = " "
            x1 = x[0:4998]
            x2 = x[4999:len(x)]
            if len(x) > 10000:
                x2 = x[4999:9998]
                x3 = x[9999: 14000]
                x4 = x[14000: 18500]
                x5 = x[18000: 22000]
                x6 = x[22000:len(x)]
                t_x3 = GoogleTranslator(source="es", target="el").translate(x3)
                t_x4 = GoogleTranslator(source="es", target="el").translate(x4)
                t_x5 = GoogleTranslator(source="es", target="el").translate(x5)
                t_x6 = GoogleTranslator(source="es", target="el").translate(x6)
            t_x1 = GoogleTranslator(source='es', target='el').translate(x1)
            t_x2 = GoogleTranslator(source='es', target='el').translate(x2)
            t = t_x1 + t_x2 + t_x3 + t_x4 + t_x5 + t_x6
            Translate.iloc[i, 0] = t
##