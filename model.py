{
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/KanishkThamman/intel-project-air_polution_predictor-/blob/main/model.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 4,
      "metadata": {
        "id": "a8ad3a7c",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "4e30c013-8126-462f-f09c-8e77606cf6f3"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Requirement already satisfied: keras-tuner in /usr/local/lib/python3.7/dist-packages (1.1.2)\n",
            "Requirement already satisfied: packaging in /usr/local/lib/python3.7/dist-packages (from keras-tuner) (21.3)\n",
            "Requirement already satisfied: numpy in /usr/local/lib/python3.7/dist-packages (from keras-tuner) (1.21.5)\n",
            "Requirement already satisfied: ipython in /usr/local/lib/python3.7/dist-packages (from keras-tuner) (5.5.0)\n",
            "Requirement already satisfied: kt-legacy in /usr/local/lib/python3.7/dist-packages (from keras-tuner) (1.0.4)\n",
            "Requirement already satisfied: requests in /usr/local/lib/python3.7/dist-packages (from keras-tuner) (2.23.0)\n",
            "Requirement already satisfied: tensorboard in /usr/local/lib/python3.7/dist-packages (from keras-tuner) (2.8.0)\n",
            "Requirement already satisfied: traitlets>=4.2 in /usr/local/lib/python3.7/dist-packages (from ipython->keras-tuner) (5.1.1)\n",
            "Requirement already satisfied: pickleshare in /usr/local/lib/python3.7/dist-packages (from ipython->keras-tuner) (0.7.5)\n",
            "Requirement already satisfied: simplegeneric>0.8 in /usr/local/lib/python3.7/dist-packages (from ipython->keras-tuner) (0.8.1)\n",
            "Requirement already satisfied: prompt-toolkit<2.0.0,>=1.0.4 in /usr/local/lib/python3.7/dist-packages (from ipython->keras-tuner) (1.0.18)\n",
            "Requirement already satisfied: decorator in /usr/local/lib/python3.7/dist-packages (from ipython->keras-tuner) (4.4.2)\n",
            "Requirement already satisfied: pygments in /usr/local/lib/python3.7/dist-packages (from ipython->keras-tuner) (2.6.1)\n",
            "Requirement already satisfied: setuptools>=18.5 in /usr/local/lib/python3.7/dist-packages (from ipython->keras-tuner) (57.4.0)\n",
            "Requirement already satisfied: pexpect in /usr/local/lib/python3.7/dist-packages (from ipython->keras-tuner) (4.8.0)\n",
            "Requirement already satisfied: six>=1.9.0 in /usr/local/lib/python3.7/dist-packages (from prompt-toolkit<2.0.0,>=1.0.4->ipython->keras-tuner) (1.15.0)\n",
            "Requirement already satisfied: wcwidth in /usr/local/lib/python3.7/dist-packages (from prompt-toolkit<2.0.0,>=1.0.4->ipython->keras-tuner) (0.2.5)\n",
            "Requirement already satisfied: pyparsing!=3.0.5,>=2.0.2 in /usr/local/lib/python3.7/dist-packages (from packaging->keras-tuner) (3.0.7)\n",
            "Requirement already satisfied: ptyprocess>=0.5 in /usr/local/lib/python3.7/dist-packages (from pexpect->ipython->keras-tuner) (0.7.0)\n",
            "Requirement already satisfied: chardet<4,>=3.0.2 in /usr/local/lib/python3.7/dist-packages (from requests->keras-tuner) (3.0.4)\n",
            "Requirement already satisfied: idna<3,>=2.5 in /usr/local/lib/python3.7/dist-packages (from requests->keras-tuner) (2.10)\n",
            "Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /usr/local/lib/python3.7/dist-packages (from requests->keras-tuner) (1.24.3)\n",
            "Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.7/dist-packages (from requests->keras-tuner) (2021.10.8)\n",
            "Requirement already satisfied: google-auth-oauthlib<0.5,>=0.4.1 in /usr/local/lib/python3.7/dist-packages (from tensorboard->keras-tuner) (0.4.6)\n",
            "Requirement already satisfied: tensorboard-data-server<0.7.0,>=0.6.0 in /usr/local/lib/python3.7/dist-packages (from tensorboard->keras-tuner) (0.6.1)\n",
            "Requirement already satisfied: protobuf>=3.6.0 in /usr/local/lib/python3.7/dist-packages (from tensorboard->keras-tuner) (3.17.3)\n",
            "Requirement already satisfied: tensorboard-plugin-wit>=1.6.0 in /usr/local/lib/python3.7/dist-packages (from tensorboard->keras-tuner) (1.8.1)\n",
            "Requirement already satisfied: wheel>=0.26 in /usr/local/lib/python3.7/dist-packages (from tensorboard->keras-tuner) (0.37.1)\n",
            "Requirement already satisfied: werkzeug>=0.11.15 in /usr/local/lib/python3.7/dist-packages (from tensorboard->keras-tuner) (1.0.1)\n",
            "Requirement already satisfied: google-auth<3,>=1.6.3 in /usr/local/lib/python3.7/dist-packages (from tensorboard->keras-tuner) (1.35.0)\n",
            "Requirement already satisfied: absl-py>=0.4 in /usr/local/lib/python3.7/dist-packages (from tensorboard->keras-tuner) (1.0.0)\n",
            "Requirement already satisfied: markdown>=2.6.8 in /usr/local/lib/python3.7/dist-packages (from tensorboard->keras-tuner) (3.3.6)\n",
            "Requirement already satisfied: grpcio>=1.24.3 in /usr/local/lib/python3.7/dist-packages (from tensorboard->keras-tuner) (1.44.0)\n",
            "Requirement already satisfied: rsa<5,>=3.1.4 in /usr/local/lib/python3.7/dist-packages (from google-auth<3,>=1.6.3->tensorboard->keras-tuner) (4.8)\n",
            "Requirement already satisfied: cachetools<5.0,>=2.0.0 in /usr/local/lib/python3.7/dist-packages (from google-auth<3,>=1.6.3->tensorboard->keras-tuner) (4.2.4)\n",
            "Requirement already satisfied: pyasn1-modules>=0.2.1 in /usr/local/lib/python3.7/dist-packages (from google-auth<3,>=1.6.3->tensorboard->keras-tuner) (0.2.8)\n",
            "Requirement already satisfied: requests-oauthlib>=0.7.0 in /usr/local/lib/python3.7/dist-packages (from google-auth-oauthlib<0.5,>=0.4.1->tensorboard->keras-tuner) (1.3.1)\n",
            "Requirement already satisfied: importlib-metadata>=4.4 in /usr/local/lib/python3.7/dist-packages (from markdown>=2.6.8->tensorboard->keras-tuner) (4.11.3)\n",
            "Requirement already satisfied: typing-extensions>=3.6.4 in /usr/local/lib/python3.7/dist-packages (from importlib-metadata>=4.4->markdown>=2.6.8->tensorboard->keras-tuner) (3.10.0.2)\n",
            "Requirement already satisfied: zipp>=0.5 in /usr/local/lib/python3.7/dist-packages (from importlib-metadata>=4.4->markdown>=2.6.8->tensorboard->keras-tuner) (3.7.0)\n",
            "Requirement already satisfied: pyasn1<0.5.0,>=0.4.6 in /usr/local/lib/python3.7/dist-packages (from pyasn1-modules>=0.2.1->google-auth<3,>=1.6.3->tensorboard->keras-tuner) (0.4.8)\n",
            "Requirement already satisfied: oauthlib>=3.0.0 in /usr/local/lib/python3.7/dist-packages (from requests-oauthlib>=0.7.0->google-auth-oauthlib<0.5,>=0.4.1->tensorboard->keras-tuner) (3.2.0)\n"
          ]
        }
      ],
      "source": [
        "!pip install keras-tuner\n",
        "\n",
        "import pandas as pd\n",
        "import numpy as np\n",
        "import matplotlib as mp\n",
        "from sklearn.preprocessing import StandardScaler\n",
        "from tensorflow.keras.layers import Dense\n",
        "from keras import Sequential\n",
        "import keras_tuner as kt\n",
        "import tensorflow as tf"
      ],
      "id": "a8ad3a7c"
    },
    {
      "cell_type": "code",
      "execution_count": 5,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "fc862488",
        "outputId": "882b8013-058c-42b7-cdd9-c4b89aed1240"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "   Unnamed: 0       City  PM2.5     CO    SO2      O3  AQI AQI_Bucket  \\\n",
            "0           0  Ahmedabad    NaN   0.92  27.64  133.36  NaN    Unknown   \n",
            "1           1  Ahmedabad    NaN   0.97  24.55   34.06  NaN    Unknown   \n",
            "2           2  Ahmedabad    NaN  17.40  29.07   30.70  NaN    Unknown   \n",
            "3           3  Ahmedabad    NaN   1.70  18.59   36.08  NaN    Unknown   \n",
            "4           4  Ahmedabad    NaN  22.10  39.33   39.31  NaN    Unknown   \n",
            "\n",
            "     State      Region    Month  Year     Season Weekday_or_weekend  \\\n",
            "0  Gujarat  5. Western  01. Jan  2015  1. Winter            Weekday   \n",
            "1  Gujarat  5. Western  01. Jan  2015  1. Winter            Weekday   \n",
            "2  Gujarat  5. Western  01. Jan  2015  1. Winter            Weekend   \n",
            "3  Gujarat  5. Western  01. Jan  2015  1. Winter            Weekend   \n",
            "4  Gujarat  5. Western  01. Jan  2015  1. Winter            Weekday   \n",
            "\n",
            "  Regular_day_or_holiday AQ_Acceptability  \n",
            "0            Regular day     Unacceptable  \n",
            "1            Regular day     Unacceptable  \n",
            "2            Regular day     Unacceptable  \n",
            "3            Regular day     Unacceptable  \n",
            "4            Regular day     Unacceptable  \n",
            "<class 'pandas.core.frame.DataFrame'>\n",
            "RangeIndex: 186670 entries, 0 to 186669\n",
            "Data columns (total 16 columns):\n",
            " #   Column                  Non-Null Count   Dtype  \n",
            "---  ------                  --------------   -----  \n",
            " 0   Unnamed: 0              186670 non-null  int64  \n",
            " 1   City                    186670 non-null  object \n",
            " 2   PM2.5                   165862 non-null  float64\n",
            " 3   CO                      182963 non-null  float64\n",
            " 4   SO2                     165241 non-null  float64\n",
            " 5   O3                      164852 non-null  float64\n",
            " 6   AQI                     165020 non-null  float64\n",
            " 7   AQI_Bucket              186670 non-null  object \n",
            " 8   State                   186670 non-null  object \n",
            " 9   Region                  186670 non-null  object \n",
            " 10  Month                   186670 non-null  object \n",
            " 11  Year                    186670 non-null  int64  \n",
            " 12  Season                  186670 non-null  object \n",
            " 13  Weekday_or_weekend      186670 non-null  object \n",
            " 14  Regular_day_or_holiday  186670 non-null  object \n",
            " 15  AQ_Acceptability        186670 non-null  object \n",
            "dtypes: float64(5), int64(2), object(9)\n",
            "memory usage: 22.8+ MB\n",
            "None\n",
            "          Unnamed: 0          PM2.5             CO            SO2  \\\n",
            "count  186670.000000  165862.000000  182963.000000  165241.000000   \n",
            "mean    93334.500000      88.805535       1.676548      13.353084   \n",
            "std     53887.131712      78.160485       3.536472      11.893876   \n",
            "min         0.000000       0.040000       0.000000       0.010000   \n",
            "25%     46667.250000      36.420000       0.630000       6.420000   \n",
            "50%     93334.500000      63.060000       1.020000      10.420000   \n",
            "75%    140001.750000     115.760000       1.550000      16.710000   \n",
            "max    186669.000000     949.990000     175.810000     193.860000   \n",
            "\n",
            "                  O3            AQI           Year  \n",
            "count  164852.000000  165020.000000  186670.000000  \n",
            "mean       41.714595     198.494382    2017.455579  \n",
            "std        24.325323     127.598226       1.595135  \n",
            "min         0.010000      13.000000    2015.000000  \n",
            "25%        25.640000      96.000000    2016.000000  \n",
            "50%        37.310000     158.000000    2018.000000  \n",
            "75%        52.620000     292.000000    2019.000000  \n",
            "max       257.730000    2049.000000    2020.000000  \n"
          ]
        }
      ],
      "source": [
        "df = pd.read_csv(\"/content/drive/MyDrive/data/Untitled spreadsheet - city_day_transformed (2) (1).csv\")\n",
        "print(df.head())\n",
        "print(df.info())\n",
        "print(df.describe())"
      ],
      "id": "fc862488"
    },
    {
      "cell_type": "code",
      "source": [
        "from google.colab import drive\n",
        "drive.mount('/content/drive')"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "QpGs0JTHKc0v",
        "outputId": "a388c1c5-1756-4026-e99f-a7fc0b1ece1d"
      },
      "id": "QpGs0JTHKc0v",
      "execution_count": 6,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount(\"/content/drive\", force_remount=True).\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 7,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "36133258",
        "outputId": "2fc2ab31-fae4-4b69-d554-f734309c1337"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "<class 'pandas.core.frame.DataFrame'>\n",
            "RangeIndex: 186670 entries, 0 to 186669\n",
            "Data columns (total 16 columns):\n",
            " #   Column                  Non-Null Count   Dtype  \n",
            "---  ------                  --------------   -----  \n",
            " 0   Unnamed: 0              186670 non-null  int64  \n",
            " 1   City                    186670 non-null  object \n",
            " 2   PM2.5                   186670 non-null  float64\n",
            " 3   CO                      186670 non-null  float64\n",
            " 4   SO2                     186670 non-null  float64\n",
            " 5   O3                      164852 non-null  float64\n",
            " 6   AQI                     186670 non-null  float64\n",
            " 7   AQI_Bucket              186670 non-null  object \n",
            " 8   State                   186670 non-null  object \n",
            " 9   Region                  186670 non-null  object \n",
            " 10  Month                   186670 non-null  object \n",
            " 11  Year                    186670 non-null  int64  \n",
            " 12  Season                  186670 non-null  object \n",
            " 13  Weekday_or_weekend      186670 non-null  object \n",
            " 14  Regular_day_or_holiday  186670 non-null  object \n",
            " 15  AQ_Acceptability        186670 non-null  object \n",
            "dtypes: float64(5), int64(2), object(9)\n",
            "memory usage: 22.8+ MB\n"
          ]
        }
      ],
      "source": [
        "df['PM2.5'] = df['PM2.5'].fillna(df['PM2.5'].mean())\n",
        "df['CO']=df['CO'].fillna(df['CO'].mean())\n",
        "df['SO2']=df['SO2'].fillna(df['SO2'].mean())\n",
        "df['AQI']=df['AQI'].fillna(df['AQI'].mean())\n",
        "df.head()\n",
        "df.info()"
      ],
      "id": "36133258"
    },
    {
      "cell_type": "code",
      "execution_count": 8,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "00c6b4ba",
        "outputId": "3d9bb830-1746-452a-e691-ae70d925f984"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "['Ahmedabad' 'Aizawl' 'Amaravati' 'Amritsar' 'Bengaluru' 'Bhopal'\n",
            " 'Brajrajnagar' 'Chandigarh' 'Chennai' 'Coimbatore' 'Delhi' 'Ernakulam'\n",
            " 'Gurugram' 'Guwahati' 'Hyderabad' 'Jaipur' 'Jorapokhar' 'Kochi' 'Kolkata'\n",
            " 'Lucknow' 'Mumbai' 'Patna' 'Shillong' 'Talcher' 'Thiruvananthapuram'\n",
            " 'Visakhapatnam']\n",
            "['Gujarat' 'Mizoram' 'Andhra Pradesh' 'Punjab' 'Karnataka'\n",
            " 'Madhya Pradesh' 'Odisha' 'Chandigarh' 'Tamil Nadu' 'Delhi' 'Kerala'\n",
            " 'Haryana' 'Assam' 'Telangana' 'Rajasthan' 'Jharkhand' 'West Bengal'\n",
            " 'Uttar Pradesh' 'Maharashtra' 'Bihar' 'Meghalaya']\n",
            "['5. Western' '2. North Eastern' '1. Northern' '6. Southern' '3. Central'\n",
            " '4. Eastern']\n",
            "['01. Jan' '02. Feb' '03. Mar' '04. Apr' '05. May' '06. Jun' '07. Jul'\n",
            " '08. Aug' '09. Sep' '10. Oct' '11. Nov' '12. Dec']\n",
            "['1. Winter' '2. Summer' '3. Monsoon' '4. Post-Monsoon']\n",
            "['Weekday' 'Weekend']\n"
          ]
        }
      ],
      "source": [
        "df.head()\n",
        "print(df['City'].unique())\n",
        "print(df['State'].unique())\n",
        "print(df['Region'].unique())\n",
        "print(df['Month'].unique())\n",
        "print(df['Season'].unique())\n",
        "print(df['Weekday_or_weekend'].unique())\n"
      ],
      "id": "00c6b4ba"
    },
    {
      "cell_type": "code",
      "execution_count": 9,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "2598f358",
        "outputId": "feffad62-1813-44c2-d275-210dfb2273da"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Delhi                 76342\n",
            "Bengaluru             20090\n",
            "Mumbai                20090\n",
            "Hyderabad             12036\n",
            "Patna                 11148\n",
            "Lucknow               10045\n",
            "Chennai                8036\n",
            "Gurugram               6716\n",
            "Kolkata                5698\n",
            "Jaipur                 3342\n",
            "Thiruvananthapuram     2224\n",
            "Ahmedabad              2009\n",
            "Visakhapatnam          1462\n",
            "Amritsar               1221\n",
            "Jorapokhar             1169\n",
            "Amaravati               951\n",
            "Brajrajnagar            938\n",
            "Talcher                 925\n",
            "Guwahati                502\n",
            "Coimbatore              386\n",
            "Shillong                310\n",
            "Chandigarh              304\n",
            "Bhopal                  289\n",
            "Ernakulam               162\n",
            "Kochi                   162\n",
            "Aizawl                  113\n",
            "Name: City, dtype: int64\n"
          ]
        }
      ],
      "source": [
        "print(df['City'].value_counts())\n",
        "\n",
        "label_encode = {\"City\": {'Ahmedabad':0,'Aizawl':1, 'Amaravati':2, 'Amritsar':3, 'Bengaluru':4, 'Bhopal':5,'Brajrajnagar':6, 'Chandigarh':7, 'Chennai':8, 'Coimbatore':9, 'Delhi':10, 'Ernakulam':11,'Gurugram':12, 'Guwahati':13, 'Hyderabad':14, 'Jaipur':15, 'Jorapokhar':16, 'Kochi':17 ,'Kolkata':18,'Lucknow':19, 'Mumbai':20 ,'Patna':21, 'Shillong':22, 'Talcher':23, 'Thiruvananthapuram':24,'Visakhapatnam':25}}\n",
        "\n",
        "df.replace(label_encode,inplace=True)"
      ],
      "id": "2598f358"
    },
    {
      "cell_type": "code",
      "execution_count": 10,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "74d1b11f",
        "outputId": "162717ec-a6fe-4291-e9f6-9ee9679af62b"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "1. Northern         97970\n",
            "6. Southern         63186\n",
            "4. Eastern          19878\n",
            "5. Western           4422\n",
            "2. North Eastern      925\n",
            "3. Central            289\n",
            "Name: Region, dtype: int64\n"
          ]
        }
      ],
      "source": [
        "print(df['Region'].value_counts())\n",
        "\n",
        "label_encode = {\"Region\": {'5. Western' :0,'2. North Eastern':1, '1. Northern':2 ,'6. Southern':3, '3. Central':4,'4. Eastern':5}}\n",
        "\n",
        "df.replace(label_encode,inplace=True)"
      ],
      "id": "74d1b11f"
    },
    {
      "cell_type": "code",
      "execution_count": 11,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "dbe5cba8",
        "outputId": "9ad6d8c1-3022-4604-b16a-d279651572d3"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Delhi             76342\n",
            "Karnataka         20090\n",
            "Maharashtra       20090\n",
            "Telangana         12036\n",
            "Bihar             11148\n",
            "Uttar Pradesh     10045\n",
            "Tamil Nadu         8422\n",
            "Haryana            6716\n",
            "West Bengal        5698\n",
            "Rajasthan          3342\n",
            "Kerala             2548\n",
            "Andhra Pradesh     2413\n",
            "Gujarat            2009\n",
            "Odisha             1863\n",
            "Punjab             1221\n",
            "Jharkhand          1169\n",
            "Assam               502\n",
            "Meghalaya           310\n",
            "Chandigarh          304\n",
            "Madhya Pradesh      289\n",
            "Mizoram             113\n",
            "Name: State, dtype: int64\n"
          ]
        }
      ],
      "source": [
        "print(df['State'].value_counts())\n",
        "\n",
        "label_encode = {\"State\": {'Gujarat':0, 'Mizoram':1 ,'Andhra Pradesh':2 ,'Punjab':3, 'Karnataka':4,'Madhya Pradesh':5, 'Odisha':6, 'Chandigarh':7, 'Tamil Nadu':8, 'Delhi':9 ,'Kerala':10,'Haryana':11, 'Assam':12 ,'Telangana':13, 'Rajasthan':14 ,'Jharkhand':15, 'West Bengal':16,'Uttar Pradesh':17, 'Maharashtra':18, 'Bihar':19, 'Meghalaya':20}}\n",
        "\n",
        "df.replace(label_encode,inplace=True)"
      ],
      "id": "dbe5cba8"
    },
    {
      "cell_type": "code",
      "execution_count": 12,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "5bd1faae",
        "outputId": "25553099-8d0c-4467-a332-5908f050d4ee"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "05. May    17360\n",
            "03. Mar    17102\n",
            "06. Jun    17080\n",
            "01. Jan    16959\n",
            "04. Apr    16718\n",
            "02. Feb    15580\n",
            "12. Dec    14638\n",
            "07. Jul    14463\n",
            "10. Oct    14446\n",
            "08. Aug    14358\n",
            "11. Nov    14003\n",
            "09. Sep    13963\n",
            "Name: Month, dtype: int64\n"
          ]
        }
      ],
      "source": [
        "print(df['Month'].value_counts())\n",
        "\n",
        "label_encode = {\"Month\": {'01. Jan':1, '02. Feb':2, '03. Mar':3, '04. Apr':4, '05. May':5, '06. Jun':6, '07. Jul':7,'08. Aug':8, '09. Sep':9 ,'10. Oct':10, '11. Nov':11, '12. Dec':12}}\n",
        "\n",
        "df.replace(label_encode,inplace=True)"
      ],
      "id": "5bd1faae"
    },
    {
      "cell_type": "code",
      "execution_count": 13,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "p6_hIbBT7tj_",
        "outputId": "e61f3505-ac53-4303-ded8-9b436b546049"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "3. Monsoon         59864\n",
            "2. Summer          51180\n",
            "4. Post-Monsoon    43087\n",
            "1. Winter          32539\n",
            "Name: Season, dtype: int64\n"
          ]
        }
      ],
      "source": [
        "print(df['Season'].value_counts())\n",
        "\n",
        "label_encode = {\"Season\": {'1. Winter':0, '2. Summer':1, '3. Monsoon':2, '4. Post-Monsoon':3}}\n",
        "\n",
        "df.replace(label_encode,inplace=True)"
      ],
      "id": "p6_hIbBT7tj_"
    },
    {
      "cell_type": "code",
      "execution_count": 14,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "4Q2fP6-q8rYE",
        "outputId": "fc73cb45-5441-4ff4-caa0-431b7a99f988"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Weekday    133348\n",
            "Weekend     53322\n",
            "Name: Weekday_or_weekend, dtype: int64\n"
          ]
        }
      ],
      "source": [
        "print(df['Weekday_or_weekend'].value_counts())\n",
        "\n",
        "label_encode = {\"Weekday_or_weekend\": {'Weekday':0, 'Weekend':1}}\n",
        "\n",
        "df.replace(label_encode,inplace=True)\n"
      ],
      "id": "4Q2fP6-q8rYE"
    },
    {
      "cell_type": "code",
      "execution_count": 15,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 288
        },
        "id": "f55ab264",
        "outputId": "6d6a483c-69dc-4317-fa71-8772c54987d0"
      },
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "   Unnamed: 0  City      PM2.5     CO    SO2      O3         AQI AQI_Bucket  \\\n",
              "0           0     0  88.805535   0.92  27.64  133.36  198.494382    Unknown   \n",
              "1           1     0  88.805535   0.97  24.55   34.06  198.494382    Unknown   \n",
              "2           2     0  88.805535  17.40  29.07   30.70  198.494382    Unknown   \n",
              "3           3     0  88.805535   1.70  18.59   36.08  198.494382    Unknown   \n",
              "4           4     0  88.805535  22.10  39.33   39.31  198.494382    Unknown   \n",
              "\n",
              "   State  Region  Month  Year  Season  Weekday_or_weekend  \\\n",
              "0      0       0      1  2015       0                   0   \n",
              "1      0       0      1  2015       0                   0   \n",
              "2      0       0      1  2015       0                   1   \n",
              "3      0       0      1  2015       0                   1   \n",
              "4      0       0      1  2015       0                   0   \n",
              "\n",
              "  Regular_day_or_holiday AQ_Acceptability  \n",
              "0            Regular day     Unacceptable  \n",
              "1            Regular day     Unacceptable  \n",
              "2            Regular day     Unacceptable  \n",
              "3            Regular day     Unacceptable  \n",
              "4            Regular day     Unacceptable  "
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-6171a956-38df-4bc0-ba38-c7d3772e3b6d\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>Unnamed: 0</th>\n",
              "      <th>City</th>\n",
              "      <th>PM2.5</th>\n",
              "      <th>CO</th>\n",
              "      <th>SO2</th>\n",
              "      <th>O3</th>\n",
              "      <th>AQI</th>\n",
              "      <th>AQI_Bucket</th>\n",
              "      <th>State</th>\n",
              "      <th>Region</th>\n",
              "      <th>Month</th>\n",
              "      <th>Year</th>\n",
              "      <th>Season</th>\n",
              "      <th>Weekday_or_weekend</th>\n",
              "      <th>Regular_day_or_holiday</th>\n",
              "      <th>AQ_Acceptability</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>88.805535</td>\n",
              "      <td>0.92</td>\n",
              "      <td>27.64</td>\n",
              "      <td>133.36</td>\n",
              "      <td>198.494382</td>\n",
              "      <td>Unknown</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>2015</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>Regular day</td>\n",
              "      <td>Unacceptable</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>88.805535</td>\n",
              "      <td>0.97</td>\n",
              "      <td>24.55</td>\n",
              "      <td>34.06</td>\n",
              "      <td>198.494382</td>\n",
              "      <td>Unknown</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>2015</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>Regular day</td>\n",
              "      <td>Unacceptable</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>2</td>\n",
              "      <td>0</td>\n",
              "      <td>88.805535</td>\n",
              "      <td>17.40</td>\n",
              "      <td>29.07</td>\n",
              "      <td>30.70</td>\n",
              "      <td>198.494382</td>\n",
              "      <td>Unknown</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>2015</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>Regular day</td>\n",
              "      <td>Unacceptable</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>3</td>\n",
              "      <td>0</td>\n",
              "      <td>88.805535</td>\n",
              "      <td>1.70</td>\n",
              "      <td>18.59</td>\n",
              "      <td>36.08</td>\n",
              "      <td>198.494382</td>\n",
              "      <td>Unknown</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>2015</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>Regular day</td>\n",
              "      <td>Unacceptable</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>4</td>\n",
              "      <td>0</td>\n",
              "      <td>88.805535</td>\n",
              "      <td>22.10</td>\n",
              "      <td>39.33</td>\n",
              "      <td>39.31</td>\n",
              "      <td>198.494382</td>\n",
              "      <td>Unknown</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>2015</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>Regular day</td>\n",
              "      <td>Unacceptable</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-6171a956-38df-4bc0-ba38-c7d3772e3b6d')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-6171a956-38df-4bc0-ba38-c7d3772e3b6d button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-6171a956-38df-4bc0-ba38-c7d3772e3b6d');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ]
          },
          "metadata": {},
          "execution_count": 15
        }
      ],
      "source": [
        "df.head()"
      ],
      "id": "f55ab264"
    },
    {
      "cell_type": "code",
      "execution_count": 41,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "9b74283b",
        "outputId": "78393a7b-d218-4003-a48d-c127e358a264"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Number of rows in x_train: 158669\n",
            "Number of rows in x_test: 28001\n",
            "Number of rows in y_train: 158669\n",
            "Number of rows in y_test: 28001\n"
          ]
        }
      ],
      "source": [
        "from sklearn.model_selection import train_test_split\n",
        "\n",
        "x_values = df[['City','State','Region','Month','Year','Season','Weekday_or_weekend']]\n",
        "y_values = df[['PM2.5','CO','SO2','AQI']]\n",
        "\n",
        "x_train, x_test, y_train, y_test = train_test_split(x_values,y_values,test_size=0.15,random_state=1)\n",
        "\n",
        "print(\"Number of rows in x_train:\", x_train.shape[0])\n",
        "print(\"Number of rows in x_test:\", x_test.shape[0])\n",
        "print(\"Number of rows in y_train:\", y_train.shape[0])\n",
        "print(\"Number of rows in y_test:\", y_test.shape[0])\n",
        "\n",
        "standardise = StandardScaler() \n",
        "x_values = standardise.fit_transform(x_values)\n",
        "x_values_df = pd.DataFrame(x_values)"
      ],
      "id": "9b74283b"
    },
    {
      "cell_type": "code",
      "execution_count": 17,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "d7bd8cf2",
        "outputId": "99f3e938-5ce8-453c-f076-cdfc7508627d"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "<class 'pandas.core.frame.DataFrame'>\n",
            "RangeIndex: 186670 entries, 0 to 186669\n",
            "Data columns (total 16 columns):\n",
            " #   Column                  Non-Null Count   Dtype  \n",
            "---  ------                  --------------   -----  \n",
            " 0   Unnamed: 0              186670 non-null  int64  \n",
            " 1   City                    186670 non-null  int64  \n",
            " 2   PM2.5                   186670 non-null  float64\n",
            " 3   CO                      186670 non-null  float64\n",
            " 4   SO2                     186670 non-null  float64\n",
            " 5   O3                      164852 non-null  float64\n",
            " 6   AQI                     186670 non-null  float64\n",
            " 7   AQI_Bucket              186670 non-null  object \n",
            " 8   State                   186670 non-null  int64  \n",
            " 9   Region                  186670 non-null  int64  \n",
            " 10  Month                   186670 non-null  int64  \n",
            " 11  Year                    186670 non-null  int64  \n",
            " 12  Season                  186670 non-null  int64  \n",
            " 13  Weekday_or_weekend      186670 non-null  int64  \n",
            " 14  Regular_day_or_holiday  186670 non-null  object \n",
            " 15  AQ_Acceptability        186670 non-null  object \n",
            "dtypes: float64(5), int64(8), object(3)\n",
            "memory usage: 22.8+ MB\n"
          ]
        }
      ],
      "source": [
        "df.info()"
      ],
      "id": "d7bd8cf2"
    },
    {
      "cell_type": "code",
      "execution_count": 18,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "000a7af2",
        "outputId": "715f367e-c3a5-48e5-a18d-657f90887679"
      },
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "Unnamed: 0                  int64\n",
              "City                        int64\n",
              "PM2.5                     float64\n",
              "CO                        float64\n",
              "SO2                       float64\n",
              "O3                        float64\n",
              "AQI                       float64\n",
              "AQI_Bucket                 object\n",
              "State                       int64\n",
              "Region                      int64\n",
              "Month                       int64\n",
              "Year                        int64\n",
              "Season                      int64\n",
              "Weekday_or_weekend          int64\n",
              "Regular_day_or_holiday     object\n",
              "AQ_Acceptability           object\n",
              "dtype: object"
            ]
          },
          "metadata": {},
          "execution_count": 18
        }
      ],
      "source": [
        "df.dtypes"
      ],
      "id": "000a7af2"
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "89044410",
        "scrolled": true,
        "outputId": "9b3aba11-dfd1-434e-f11b-c2bd409971dc"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Trial 3 Complete [00h 02m 22s]\n",
            "accuracy: 0.006844437215477228\n",
            "\n",
            "Best accuracy So Far: 0.992979109287262\n",
            "Total elapsed time: 00h 05m 02s\n",
            "\n",
            "Search: Running Trial #4\n",
            "\n",
            "Value             |Best Value So Far |Hyperparameter\n",
            "406               |286               |units1\n",
            "454               |442               |units2\n",
            "502               |226               |units3\n",
            "\n",
            "Epoch 1/5\n",
            "4959/4959 [==============================] - 24s 5ms/step - loss: 1673.3994 - accuracy: 0.9930 - val_loss: 1674.9119 - val_accuracy: 0.9932\n",
            "Epoch 2/5\n",
            "4959/4959 [==============================] - 24s 5ms/step - loss: 1673.3972 - accuracy: 0.9930 - val_loss: 1674.9119 - val_accuracy: 0.9932\n",
            "Epoch 3/5\n",
            "4959/4959 [==============================] - 24s 5ms/step - loss: 1673.3989 - accuracy: 0.9930 - val_loss: 1674.9119 - val_accuracy: 0.9932\n",
            "Epoch 4/5\n",
            "4959/4959 [==============================] - 24s 5ms/step - loss: 1673.3978 - accuracy: 0.9930 - val_loss: 1674.9119 - val_accuracy: 0.9932\n",
            "Epoch 5/5\n",
            "4925/4959 [============================>.] - ETA: 0s - loss: 1673.0494 - accuracy: 0.9930"
          ]
        }
      ],
      "source": [
        "\n",
        "def build_model(hp):\n",
        "  model = Sequential()\n",
        "  hp_units1 = hp.Int('units1', min_value=7, max_value=512, step=12)\n",
        "  hp_units2 = hp.Int('units2', min_value=10, max_value=512, step=12)\n",
        "  hp_units3 = hp.Int('units3', min_value=10, max_value=512, step=12)\n",
        "  model.add(Dense(units=hp_units1, activation='relu'))\n",
        "  model.add(tf.keras.layers.Dense(units=hp_units2, activation='relu'))\n",
        "  model.add(tf.keras.layers.Dense(units=hp_units3, activation='relu'))\n",
        "  model.add(Dense(4, kernel_initializer='normal', activation='relu'))\n",
        "  model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])\n",
        "  return model\n",
        "\n",
        "tuner = kt.RandomSearch(\n",
        "    build_model,\n",
        "    objective='accuracy',\n",
        "    max_trials=50)\n",
        "tuner.search(x_train, y_train, epochs=5, validation_data=(x_test, y_test))\n",
        "best_model = tuner.get_best_models()[0]"
      ],
      "id": "89044410"
    },
    {
      "cell_type": "code",
      "source": [
        ""
      ],
      "metadata": {
        "id": "EbQntayPVRCv"
      },
      "id": "EbQntayPVRCv",
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        " tf.config.list_physical_devices('GPU') "
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "irjSsaMPSyzA",
        "outputId": "31b7a451-e6b2-45f7-b0d7-faecdb2d9666"
      },
      "id": "irjSsaMPSyzA",
      "execution_count": 36,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]"
            ]
          },
          "metadata": {},
          "execution_count": 36
        }
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 33,
      "metadata": {
        "id": "41b41284"
      },
      "outputs": [],
      "source": [
        "tf.debugging.set_log_device_placement(True)\n",
        "tf.ones([])\n",
        "# [...] op Fill in device /job:localhost/replica:0/task:0/device:GPU:0\n",
        "with tf.device(\"CPU\"):\n",
        " tf.ones([])\n",
        "# [...] op Fill in device /job:localhost/replica:0/task:0/device:CPU:0\n",
        "tf.debugging.set_log_device_placement(False)"
      ],
      "id": "41b41284"
    }
  ],
  "metadata": {
    "colab": {
      "name": "AI_Bootcamp.ipynb",
      "provenance": [],
      "collapsed_sections": [],
      "include_colab_link": true
    },
    "kernelspec": {
      "display_name": "Python 3 (ipykernel)",
      "language": "python",
      "name": "python3"
    },
    "language_info": {
      "codemirror_mode": {
        "name": "ipython",
        "version": 3
      },
      "file_extension": ".py",
      "mimetype": "text/x-python",
      "name": "python",
      "nbconvert_exporter": "python",
      "pygments_lexer": "ipython3",
      "version": "3.9.7"
    },
    "accelerator": "GPU"
  },
  "nbformat": 4,
  "nbformat_minor": 5
}