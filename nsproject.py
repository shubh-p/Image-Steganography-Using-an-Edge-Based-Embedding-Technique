import cv2  # for reading and writing image data
import heapq  # for heap data structure
import os  # to work with files
import random  # for randomisation
from AES import *

compTextRef = ""
side0Mode = 0
side1Mode = 0
side2Mode = 0
side3Mode = 0
embeddedBits = 0

retrievedBits = 0
retrievedTextRef = ""
retrievedTextLength = 0
lengthRetrieved = False


def make_frequency_dict(text):
    frequency = {}
    for character in text:
        if character not in frequency:
            frequency[character] = 0
        frequency[character] += 1
    return frequency


def pad_encoded_text(encoded_text):
    extra_padding = 120 - len(encoded_text) % 128
    for i1 in range(extra_padding):
        encoded_text += "0"

    padded_info = "{0:08b}".format(extra_padding)
    encoded_text = padded_info + encoded_text
    return encoded_text


def get_byte_array(padded_encoded_text):
    if len(padded_encoded_text) % 8 != 0:
        print("Error occurred while padding")
        exit(0)
    b = bytearray()
    for i1 in range(0, len(padded_encoded_text), 8):
        byte1 = padded_encoded_text[i1:i1 + 8]
        b.append(int(byte1, 2))
    return b


def remove_padding(padded_encoded_text):
    padded_info = padded_encoded_text[:8]
    extra_padding = int(padded_info, 2)
    padded_encoded_text = padded_encoded_text[8:]
    encoded_text = padded_encoded_text[:-1 * extra_padding]

    return encoded_text


class HuffmanCoding:
    def __init__(self, path):
        self.path = path
        self.heap = []
        self.codes = {}
        self.reverse_mapping = {}

    # noinspection PyTypeChecker
    class HeapNode:
        def __init__(self, char, freq):
            self.char = char
            self.freq = freq
            self.left = None
            self.right = None

        # defining comparators less_than and equals
        def __lt__(self, other):
            return self.freq < other.freq

        def __eq__(self, other):
            if other is None:
                return False
            if not isinstance(other, self):
                return False
            return self.freq == other

    # functions for compression
    def make_heap(self, frequency):
        for key in frequency:
            node = self.HeapNode(key, frequency[key])
            heapq.heappush(self.heap, node)

    def merge_nodes(self):
        while len(self.heap) > 1:
            node1 = heapq.heappop(self.heap)
            node2 = heapq.heappop(self.heap)

            merged = self.HeapNode(None, node1.freq + node2.freq)
            merged.left = node1
            merged.right = node2

            heapq.heappush(self.heap, merged)

    def make_codes_helper(self, root, current_code):
        if root is None:
            return

        if root.char is not None:
            self.codes[root.char] = current_code
            self.reverse_mapping[current_code] = root.char
            return

        self.make_codes_helper(root.left, current_code + "0")
        self.make_codes_helper(root.right, current_code + "1")

    def make_codes(self):
        root = heapq.heappop(self.heap)
        current_code = ""
        self.make_codes_helper(root, current_code)

    def get_encoded_text(self, text):
        encoded_text = ""
        for character in text:
            encoded_text += self.codes[character]
        return encoded_text

    def compress(self):
        global compTextRef
        filename, file_extension = os.path.splitext(self.path)
        outputPath = filename + ".bin"

        with open(self.path, 'r+') as file1, open(outputPath, 'wb') as output:
            text = file1.read()
            text = text.rstrip()
            print("Bits in input text: ", len(text) * 8, "bits")
            frequency = make_frequency_dict(text)
            self.make_heap(frequency)
            self.merge_nodes()
            self.make_codes()

            encoded_text = self.get_encoded_text(text)
            padded_encoded_text = pad_encoded_text(encoded_text)

            padded_encoded_text = padded_encoded_text
            compTextRef = padded_encoded_text
            b = get_byte_array(padded_encoded_text)
            output.write(bytes(b))
        return outputPath

    # functions for decompression
    def decode_text(self, encoded_text):
        current_code = ""
        decoded_text = ""

        for bit in encoded_text:
            current_code += bit
            if current_code in self.reverse_mapping:
                character = self.reverse_mapping[current_code]
                decoded_text += character
                current_code = ""

        return decoded_text

    def decompress(self, input_path):
        filename, file_extension = os.path.splitext(self.path)
        outputPath = filename + "_decompressed" + ".txt"
        with open(input_path, 'rb') as file1, open(outputPath, 'w') as output:
            bit_string = ""
            byte1 = file1.read(1)
            while len(byte1) > 0:
                byte1 = ord(byte1)
                bits1 = bin(byte1)[2:].rjust(8, '0')
                bit_string += bits1
                byte1 = file1.read(1)
            encoded_text = remove_padding(bit_string)
            decompressed_text = self.decode_text(encoded_text)
            output.write(decompressed_text)
        return outputPath


def randomiseSides():
    n = []
    count1 = 0
    keepLooping = True
    sideData = []
    while keepLooping:  # randomise 3 sides
        temp1 = random.randint(0, 3)
        if temp1 not in n:
            n.append(temp1)
            count1 += 1
        if count1 == 3:
            keepLooping = False
    for i1 in n:
        mode = random.randint(0, 1)
        if i1 == 0:
            sideData.append([0, 0, mode])
        elif i1 == 1:
            sideData.append([0, 1, mode])
        elif i1 == 2:
            sideData.append([1, 0, mode])
        else:
            sideData.append([1, 1, mode])
    return sideData


def getBinary(x1):
    s = bin(x1)[2:]
    while len(s) < 8:
        s = '0' + s
    return s


def getDecimal(s):
    a = (int(s[0], 2), int(s[1], 2), int(s[2], 2))
    return a


def changePixel(row, col, data):
    global img
    (b, g, r) = img[row][col]
    s = [getBinary(b), getBinary(g), getBinary(r)]
    for i1 in range(3):
        temp1 = list(s[i1])
        temp1[7] = str(data[i1])
        s[i1] = "".join(temp1)
    img[row][col] = getDecimal(s)


def assignSideInfo(currRow, sideData):
    global rows, columns
    changePixel(currRow, currRow, sideData[0])
    changePixel(currRow, columns - currRow - 1, sideData[1])
    changePixel(rows - currRow - 1, columns - currRow - 1, sideData[2])


def assignMode(side, mode):
    global side0Mode, side1Mode, side2Mode, side3Mode
    if side == 0:
        side0Mode = mode
    elif side == 1:
        side1Mode = mode
    elif side == 2:
        side2Mode = mode
    else:
        side3Mode = mode


def side0(pixel, channel, data, currRow):
    global columns, img
    tRow = currRow
    if side0Mode == 0:
        tCol = currRow + 1 + pixel
    else:
        tCol = columns - currRow - 2 - pixel
    temp1 = img[tRow][tCol][channel] % 2
    if temp1 != int(data):
        if temp1 == 0:
            img[tRow][tCol][channel] += 1
        else:
            img[tRow][tCol][channel] -= 1


def side1(pixel, channel, data, currRow):
    global rows, columns, img
    tCol = columns - currRow - 1
    if side1Mode == 0:
        tRow = currRow + 1 + pixel
    else:
        tRow = rows - currRow - 2 - pixel
    temp1 = img[tRow][tCol][channel] % 2
    if temp1 != int(data):
        if temp1 == 0:
            img[tRow][tCol][channel] += 1
        else:
            img[tRow][tCol][channel] -= 1


def side2(pixel, channel, data, currRow):
    global rows, columns, img
    tRow = rows - currRow - 1
    if side2Mode == 0:
        tCol = columns - currRow - 2 - pixel
    else:
        tCol = currRow + 1 + pixel
    temp1 = img[tRow][tCol][channel] % 2
    if temp1 != int(data):
        if temp1 == 0:
            img[tRow][tCol][channel] += 1
        else:
            img[tRow][tCol][channel] -= 1


def side3(pixel, channel, data, currRow):
    global rows, img
    tCol = currRow
    if side3Mode == 0:
        tRow = rows - currRow - 2 - pixel
    else:
        tRow = currRow + 1 + pixel
    temp1 = img[tRow][tCol][channel] % 2
    if temp1 != int(data):
        if temp1 == 0:
            img[tRow][tCol][channel] += 1
        else:
            img[tRow][tCol][channel] -= 1


def embedEdge(currRow):
    global compTextRef, embeddedBits, rows, img

    sides = []
    bitsThisIter = 0

    (b, g, r) = img[currRow, currRow]
    s = [getBinary(b), getBinary(g), getBinary(r)]
    side = int(s[0][7] + s[1][7], 2)
    sides.append(side)
    assignMode(side, int(s[2][7]))

    (b, g, r) = img[currRow, columns - currRow - 1]
    s = [getBinary(b), getBinary(g), getBinary(r)]
    side = int(s[0][7] + s[1][7], 2)
    sides.append(side)
    assignMode(side, int(s[2][7]))

    (b, g, r) = img[rows - currRow - 1, columns - currRow - 1]
    s = [getBinary(b), getBinary(g), getBinary(r)]
    side = int(s[0][7] + s[1][7], 2)
    sides.append(side)
    assignMode(side, int(s[2][7]))

    for i1 in range(4):
        if i1 not in sides:
            sides.append(i1)
            break

    print("Sides embedding order:", sides)
    print("Modes of side 0 to 3:", side0Mode, side1Mode, side2Mode, side3Mode)

    spaceAvailable = (rows - (currRow + 1) * 2) * 3 * 4
    while True:
        if spaceAvailable < 64:
            if spaceAvailable >= len(compTextRef) - embeddedBits:
                n = spaceAvailable // 4
            else:
                return True
        else:
            n = 16
        for i1 in range(n):
            for j1 in range(4):
                if embeddedBits < len(compTextRef):
                    temp1 = bitsThisIter // 4
                    if sides[j1] == 0:
                        side0(temp1 // 3, temp1 % 3, compTextRef[embeddedBits], currRow)
                    elif sides[j1] == 1:
                        side1(temp1 // 3, temp1 % 3, compTextRef[embeddedBits], currRow)
                    elif sides[j1] == 2:
                        side2(temp1 // 3, temp1 % 3, compTextRef[embeddedBits], currRow)
                    else:
                        side3(temp1 // 3, temp1 % 3, compTextRef[embeddedBits], currRow)
                    embeddedBits += 1
                    bitsThisIter += 1
                    spaceAvailable -= 1
                else:
                    return False


def embed():
    currRow = 0
    keepLooping = True
    while keepLooping:
        sideData = randomiseSides()
        assignSideInfo(currRow, sideData)
        keepLooping = embedEdge(currRow)
        if embeddedBits >= len(compTextRef):
            keepLooping = False
        currRow += 1
        print('Edge', currRow, 'filled')
    print('Bits embedded:', embeddedBits, 'bits')


def getSide(a, b):
    if a == 0:
        if b == 0:
            return 0
        else:
            return 1
    else:
        if b == 0:
            return 2
        else:
            return 3


def getSideData(currRow):
    global img, side0Mode, side1Mode, side2Mode, side3Mode, columns, rows
    sides = []

    (b, g, r) = img[currRow][currRow]
    side = getSide(b % 2, g % 2)
    sides.append(side)
    assignMode(side, r % 2)
    (b, g, r) = img[currRow][columns - currRow - 1]
    side = getSide(b % 2, g % 2)
    sides.append(side)
    assignMode(side, r % 2)
    (b, g, r) = img[rows - currRow - 1][columns - currRow - 1]
    side = getSide(b % 2, g % 2)
    sides.append(side)
    assignMode(side, r % 2)

    for i1 in range(4):
        if i1 not in sides:
            sides.append(i1)
            assignMode(i1, 0)

    return sides


def getSide0(currRow, pixel, channel):
    global img, columns
    tRow = currRow
    if side0Mode == 0:
        tCol = currRow + pixel + 1
    else:
        tCol = columns - currRow - 2 - pixel
    return img[tRow][tCol][channel] % 2


def getSide1(currRow, pixel, channel):
    global img, columns, rows
    tCol = columns - currRow - 1
    if side1Mode == 0:
        tRow = currRow + 1 + pixel
    else:
        tRow = rows - currRow - 2 - pixel
    return img[tRow][tCol][channel] % 2


def getSide2(currRow, pixel, channel):
    global img, rows, columns
    tRow = rows - currRow - 1
    if side2Mode == 0:
        tCol = columns - currRow - 2 - pixel
    else:
        tCol = currRow + 1 + pixel
    return img[tRow][tCol][channel] % 2


def getSide3(currRow, pixel, channel):
    global img, rows
    tCol = currRow
    if side3Mode == 0:
        tRow = rows - currRow - 2 - pixel
    else:
        tRow = currRow + 1 + pixel
    return img[tRow][tCol][channel] % 2


def retrieveData(currRow, sides):
    global img, side0Mode, side1Mode, side2Mode, side3Mode,\
        retrievedTextRef, lengthRetrieved, retrievedTextLength, retrievedBits

    totalBits = (rows - (currRow + 1) * 2) * 3 * 4
    bitsThisIter = 0

    while True:
        for i1 in range(4):
            for j1 in range(4):
                temp1 = bitsThisIter // 4
                if sides[j1] == 0:
                    retrievedTextRef += str(getSide0(currRow, temp1 // 3, temp1 % 3))
                elif sides[j1] == 1:
                    retrievedTextRef += str(getSide1(currRow, temp1 // 3, temp1 % 3))
                elif sides[j1] == 2:
                    retrievedTextRef += str(getSide2(currRow, temp1 // 3, temp1 % 3))
                else:
                    retrievedTextRef += str(getSide3(currRow, temp1 // 3, temp1 % 3))
                retrievedBits += 1
                bitsThisIter += 1
                totalBits -= 1
                if retrievedBits == 16 and retrievedTextLength == 0:
                    retrievedTextLength = int(retrievedTextRef, 2)
                    retrievedTextRef = ""
                if retrievedTextLength != 0 and retrievedBits - 16 == retrievedTextLength:
                    return False
                if totalBits == 0:
                    return True


def retrieve():
    global img
    currRow = 0
    keepLooping = True
    while keepLooping:
        sides = getSideData(currRow)
        print("Sides retrieval order:", sides)
        print("Modes of side 0 to 3:", side0Mode, side1Mode, side2Mode, side3Mode)
        keepLooping = retrieveData(currRow, sides)
        currRow += 1
        print('Edge', currRow, 'retrieved')
    print("Retrieved bits:", retrievedBits, "bits")


print("BCI 3005 Digital Watermarking and Steganography  Project")
print("Title: Image Steganography Using a LSB Based Embedding Technique")
print("Members:\n18BCI0268\n20MID0175\n20BKT0056")
ipPath = input("\nEnter the name of file containing input text: ")
imPath = input("Enter the name of file containing the image: ")
img = cv2.imread(imPath, 1)
print("\nIn image:\nRows:", len(img), "\nColumns:", len(img[0]))


print("\nStep 1: Text compression")
h = HuffmanCoding(ipPath)
output_path = h.compress()
print("Compressed file path: " + output_path)
print("Compressed text in binary: " + compTextRef)
print("Bits in compressed text:", len(compTextRef), "bits")
print("Text compression completed")


print("\nStep 2: Encrypting using AES")
encrypted = ""
plainValues = []
with open(output_path, 'rb') as file:
    encryptionIp = ""
    byte = file.read(1)
    while len(byte) > 0:
        byte = ord(byte)
        bits = bin(byte)[2:].rjust(8, '0')
        encryptionIp += bits
        byte = file.read(1)
file.close()
len1 = len(encryptionIp) / 4
encryptionIpCopy = encryptionIp
encryptionIp = hex(int(encryptionIp, 2))[2:].zfill(int(len1))
for j in range(int(len(encryptionIp) / 32)):
    input_plain = encryptionIp[j:j+32]
    input_plain = input_plain.rstrip().replace(" ", "").lower()
    plain = split_string(8, input_plain)
    plain = [split_string(2, word) for word in plain]
    plainValues.append(plain)
    rounds = [apply_round_key(keys[0], plain)]
    for i in range(1, 10 + 1):
        rounds.append(aes_round(rounds[i - 1], keys[i]))
    paddingLen = len("{}".format(''.join(int_to_hex(flatten(rounds[10]))))) * 4
    temp = bin(int("{}".format(''.join(int_to_hex(flatten(rounds[10])))), 16))[2:].zfill(paddingLen)
    encrypted += temp
print("After encryption (in binary): " + encrypted)
print("Encryption completed")


print("\nStep 3: Embedding data into image")
rows = len(img)
columns = len(img[0])
compTextRef = encrypted
compTextLen = bin(len(compTextRef))
compTextLen = compTextLen[2:]
while len(compTextLen) < 16:
    compTextLen = '0' + compTextLen
compTextRef = compTextLen + compTextRef
print("Size of data to be embedded after padding 16 bit length:", len(compTextRef), "bits")
embed()
cv2.imwrite('encrypted.PNG', img)
cv2.imshow('encrypted image', img)
print("Data embedded successfully")


print("\nStep 4: Retrieving data from image")
img = cv2.imread('encrypted.PNG', 1)
rows = len(img)
columns = len(img[0])
retrieve()
print("Bits in retrieved text after removing 16 bit padding:", len(retrievedTextRef), "bits")
print("Retrieved text: " + retrievedTextRef)
print("Successfully retrieved")


print("\nStep 5: Decrypting retrieved data")
decrypted = ""
len1 = len(retrievedTextRef) / 4
retrievedTextRef = hex(int(retrievedTextRef, 2))[2:].zfill(int(len1))
for j in range(int(len(retrievedTextRef) / 32)):
    plain = plainValues[j]
    for i in range(10, 0, -1):
        rounds.append(inverse_aes_round(rounds[i - 1], keys[i]))
    paddingLen = len("{}".format(''.join(flatten(plainValues[j])))) * 4
    temp = bin(int("{}".format(''.join(flatten(plainValues[j]))), 16))[2:].zfill(paddingLen)
    decrypted += temp
print("Number of bits in decrypted data:", len(decrypted), "bits")
print("Decrypted data: " + decrypted)
print("Decryption completed")


print("\nStep 6: Decompression of data")
print("Decompressed file path: " + h.decompress(output_path))
print("Decompression completed")
cv2.waitKey(0)
