#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <ctype.h>
#include <stdbool.h>
#include <math.h>

// -------- Global Variables (Updated for multi-file support) --------
char inputFilePath1[256];  // File 1 path
char inputFilePath2[256];  // File 2 path
char toxicWordsFile[256];  // Unused, placeholder
char stopWordsFile[256];   // Unused, placeholder
char outputFilePath[256];

char words1[30000][50];     // File 1 word list
char words2[30000][50];     // File 2 word list
char toxicWords[500][50];   // Unused, placeholder

int  wordCount1 = 0;       // File 1 word count
int  wordCount2 = 0;       // File 2 word count
int  toxicCount = 0;

bool file1Loaded = false;  // Whether file 1 is loaded
bool file2Loaded = false;  // Whether file 2 is loaded

// ====== New: Advanced Text Analysis Structure ======
#define MAX_WORDS 40000
#define MAX_WORD_LENGTH 50
#define MAX_STOPWORDS 500
#define MAX_TEXT_LENGTH 50000

struct WordInfo {
    char word[MAX_WORD_LENGTH];
    int count;
};

// -------- Function Declarations (Updated) --------
void loadTextFile(int fileNumber);
void displayGeneralStatistics();
void displayToxicWordAnalysis();
void sortAndDisplayTopNWords();
void saveResultsToFile();
void loadDictionaries();
void handleError(const char* message);
void translateTextToEnglish();
void displayNegativityScale();
void displayWordOccurrence();
void showFileHistory(int fileNumber);
void handleFileMenu();
bool isCSVFile(const char* filename);
void processCSVFile(FILE* f, char (*targetWords)[50], int* targetWordCount);
bool isFileCorrupted(const char* filePath, const char* fileContent, size_t contentSize); // New: File corruption detection

// ====== New: Advanced Text Analysis Function Declarations ======
int load_stopwords(char stopwords[][MAX_WORD_LENGTH]);
int is_stopword(char* word, char stopwords[][MAX_WORD_LENGTH], int stop_count);
int tokenize_and_clean(char* text, struct WordInfo words[], char stopwords[][MAX_WORD_LENGTH], int stop_count, int* total_chars);
int count_sentences(char* text);
void generate_statistics(struct WordInfo words[], int word_count, int total_words, int total_chars, int sentences, int stopwords_removed);
void sort_by_frequency(struct WordInfo words[], int count);
void show_top_words(struct WordInfo words[], int word_count, int n);
void advancedTextAnalysis();

// ====== New: General Utilities (Ready to use) ======
// Read a whole line (including spaces) and remove trailing \n\r
static int read_line(char* buf, size_t cap) {
    if (!fgets(buf, cap, stdin)) return 0;
    size_t n = strlen(buf);
    while (n && (buf[n - 1] == '\n' || buf[n - 1] == '\r')) buf[--n] = '\0';
    return 1;
}

// Allow input paths with quotes (common when copying from file explorer), auto-remove them
static void strip_quotes(char* s) {
    size_t n = strlen(s);
    if (n >= 2 && s[0] == '"' && s[n - 1] == '"') {
        memmove(s, s + 1, n - 2);
        s[n - 2] = '\0';
    }
}

// Normalize a line: lowercase + non-alphanumeric to space
static void normalize_line(char* s) {
    for (char* p = s; *p; ++p) {
        unsigned char c = (unsigned char)*p;
        if (isalnum(c)) *p = (char)tolower(c);
        else *p = ' ';
    }
}

// --- Windows UTF-8 Path Support (More stable for Chinese/non-ASCII) ---
#ifdef _WIN32
#include <windows.h>
#include <wchar.h>
static FILE* fopen_u8(const char* utf8Path, const char* mode) {
    wchar_t wpath[MAX_PATH], wmode[8];
    if (!MultiByteToWideChar(CP_UTF8, 0, utf8Path, -1, wpath, MAX_PATH)) return NULL;
    if (!MultiByteToWideChar(CP_UTF8, 0, mode, -1, wmode, 8)) return NULL;
    return _wfopen(wpath, wmode);
}
#endif

// Unified read-only open: auto-remove quotes, Windows uses wide characters
static FILE* open_file_read(const char* pathIn) {
    static char tmp[1024];
    strncpy(tmp, pathIn, sizeof(tmp) - 1);
    tmp[sizeof(tmp) - 1] = '\0';
    strip_quotes(tmp);
#ifdef _WIN32
    FILE* f = fopen_u8(tmp, "r");
#else
    FILE* f = fopen(tmp, "r");
#endif
    if (!f) {
        fprintf(stderr, "[!] fopen failed for: %s\n", tmp);
        perror("reason");
    }
    return f;
}

//file corruption detection
bool isFileCorrupted(const char* filePath, const char* fileContent, size_t contentSize) {
    if (contentSize==0) {
        printf("[!] Error : File is empty\n");
        return false; // Empty file is not considered corrupted, just no content
    }
    
    int totalChars = 0;
    int printableChars = 0;
    int controlChars = 0;
    int consecutiveNonsense = 0;
    int maxConsecutiveNonsense = 0;
    int wordLikeSequences = 0;
    
    // Analyze file content characteristics
    for (size_t i = 0; i < contentSize; i++) {
        unsigned char c = (unsigned char)fileContent[i];
        totalChars++;
        
        // Printable characters (including spaces, punctuation)
        if (isprint(c) || c == '\t' || c == '\n' || c == '\r') {
            printableChars++;
            consecutiveNonsense = 0;
            
            // Detect word-like sequences (letter sequences)
            if (isalpha(c)) {
                // Check if this is the start of a word
                if (i == 0 || !isalpha((unsigned char)fileContent[i-1])) {
                    // Check the length of this letter sequence
                    size_t j = i;
                    int wordLen = 0;
                    while (j < contentSize && isalpha((unsigned char)fileContent[j])) {
                        wordLen++;
                        j++;
                    }
                    if (wordLen >= 2 && wordLen <= 20) { // Reasonable word length
                        wordLikeSequences++;
                    }
                }
            }
        } else {
            // Control characters or non-printable characters
            controlChars++;
            consecutiveNonsense++;
            if (consecutiveNonsense > maxConsecutiveNonsense) {
                maxConsecutiveNonsense = consecutiveNonsense;
            }
        }
    }
    
    // Calculate various ratios
    double printableRatio = (double)printableChars / totalChars;
    double controlCharRatio = (double)controlChars / totalChars;
    double wordDensity = (double)wordLikeSequences / (contentSize / 100.0); // Words per 100 characters
    
    printf("[i] File analysis: %zu chars, %.1f%% printable, %.1f%% control, word density: %.1f/100chars\n", contentSize, printableRatio * 100, controlCharRatio * 100, wordDensity);
    
    // Corrupted file detection rules
    bool likelyCorrupted = false;
    const char* reason = NULL;
    
    if (printableRatio < 0.60) { //less than 60% printable characters
        likelyCorrupted = true;
        reason = "low percentage of printable characters";
    } else if (controlCharRatio > 0.40) { //more than 40% control characters
        likelyCorrupted = true;
        reason = "high percentage of control characters";
    } else if (maxConsecutiveNonsense > 10) { //more than 10 consecutive nonsense characters
        likelyCorrupted = true;
        reason = "long sequences of nonsense characters";
    } else if (wordDensity < 0.5 && contentSize > 100) { //extremely low word density in large files
        likelyCorrupted = true;
        reason = "extremely low word density";
    }
    
    if (likelyCorrupted) {
        printf("[X] Error: File you loaded seems to be corrupted. Reason: %s\n", reason);
        printf("[X] The file may contain nonsense characters.\n");
        return true; //stop loading if file is corrupted
    }
    
    return false; //file is normal 
}

//main program + menu interface
int main() {
    int userChoice;

    for (;;) {
        printf("\nToxic Word Text Analyser Menu\n");
        printf("1. Load text file for analysis (e.g., enter text file path)\n");
        printf("2. Display general word statistics (e.g., word counts, frequencies)\n");
        printf("3. Display toxic word analysis (e.g., toxicity score)\n");
        printf("4. Sort and display top N words (e.g., by frequency or toxicity)\n");
        printf("5. Save results to output file\n");
        printf("6. Advanced Text Analysis (with stopwords filtering)\n");
        printf("7. Translate content to English\n");
        printf("8. Exit System\n");
        printf("Type in a choice or type 8 to exit the system: ");

        if (scanf("%d", &userChoice) != 1) {
            // Clear invalid input
            int c; while ((c = getchar()) != '\n' && c != EOF);
            printf("Invalid input. Please enter a number 1-8.\n");
            continue;
        }
        // Consume trailing newline for read_line to work properly
        int c; while ((c = getchar()) != '\n' && c != EOF);

        switch (userChoice) {
        case 1:
            handleFileMenu();
            break;

        case 2:
            displayGeneralStatistics();
            break;

        case 3:
        {
            if (!file1Loaded && !file2Loaded) {
                printf("[!] No text files loaded. Use menu 1 first.\n");
                break;
            }
            for (;;) {
                char subChoiceStr[10];
                printf("Toxic Word Analysis Sub-Menu:\n");
                printf("3.1 Display negativity scale (1-5)\n");
                printf("3.2 Display word occurrence\n");
                printf("3.3 Exit to main menu\n");
                printf("Enter sub-choice (3.1, 3.2, or 3.3): ");
                if (!read_line(subChoiceStr, sizeof(subChoiceStr))) {
                    printf("Failed to read sub-choice.\n");
                    break;
                }
                if (strcmp(subChoiceStr, "3.1") == 0) {
                    displayNegativityScale();
                }
                else if (strcmp(subChoiceStr, "3.2") == 0) {
                    displayWordOccurrence();
                }
                else if (strcmp(subChoiceStr, "3.3") == 0) {
                    break;  // Exit sub-menu
                }
                else {
                    printf("Invalid sub-choice. Please enter 3.1, 3.2, or 3.3.\n");
                }
            }
        }
        break;

        case 4:
            sortAndDisplayTopNWords();
            break;

        case 5:
            printf("Enter output file path (e.g., analysis_report.txt):\n> ");
            if (!read_line(outputFilePath, sizeof(outputFilePath))) {
                handleError("Failed to read output path"); break;
            }
            saveResultsToFile();
            break;

        case 6:
            advancedTextAnalysis();
            break;

        case 7:
            translateTextToEnglish();
            break;

        case 8:
            printf("Exiting the system... Goodbye!\n");
            return 0;  // Actually exit

        default:
            printf("%d is an invalid choice. Please enter a number between 1 and 8.\n", userChoice);
        }
    }
}

// ====== Updated: Load Text File Function (Supports CSV and corrupted file detection) ======
void loadTextFile(int fileNumber) {
    char* filePath = (fileNumber == 1) ? inputFilePath1 : inputFilePath2;
    char (*targetWords)[50] = (fileNumber == 1) ? words1 : words2;
    int* targetWordCount = (fileNumber == 1) ? &wordCount1 : &wordCount2;
    bool* targetFileLoaded = (fileNumber == 1) ? &file1Loaded : &file2Loaded;

    FILE* f = open_file_read(filePath);
    if (!f) {
        handleError("Cannot open input file");
        return;
    }

    *targetWordCount = 0;

    // Remove quotes for file type detection
    char cleanPath[256];
    strncpy(cleanPath, filePath, sizeof(cleanPath) - 1);
    cleanPath[sizeof(cleanPath) - 1] = '\0';
    strip_quotes(cleanPath);

    // ====== New: File content pre-scan and corruption detection ======
    printf("[i] Pre-scanning file for corruption detection...\n");
    
    // Read entire file content for detection
    fseek(f, 0, SEEK_END);
    long fileSize = ftell(f);
    fseek(f, 0, SEEK_SET);
    
    if (fileSize > 0) {
        char* fileContent = (char*)malloc(fileSize + 1);
        if (fileContent) {
            size_t bytesRead = fread(fileContent, 1, fileSize, f);
            fileContent[bytesRead] = '\0';
            
            // Detect if file is corrupted
            if (isFileCorrupted(cleanPath, fileContent, bytesRead)) {
                free(fileContent);
                fclose(f);
                *targetFileLoaded = false;
                return; // Stop loading corrupted file
            }
            
            free(fileContent);
        }
        // Reset file pointer to beginning
        fseek(f, 0, SEEK_SET);
    }

    // Check file type and process accordingly
    if (isCSVFile(cleanPath)) {
        printf("[i] Detected CSV file format - now converting columns to text...\n");
        processCSVFile(f, targetWords, targetWordCount);
        printf("[✓] CSV file converted to text format\n");
    }
    else {
        printf("[i] Detected text file format - processing as text...\n");
        // Original text file processing logic
        char line[4096];
        while (fgets(line, sizeof(line), f)) {
            normalize_line(line); // Lowercase + non-alphanumeric → space
            char* tok = strtok(line, " \t\r\n");
            while (tok) {
                size_t L = strlen(tok);
                if (L > 0 && L < sizeof(targetWords[0])) {
                    if (*targetWordCount < 30000) {
                        strcpy(targetWords[*targetWordCount], tok);
                        (*targetWordCount)++;
                    }
                    else {
                        // Stop if capacity exceeded
                        printf("[!] Warning: Reached maximum token limit (30000)\n");
                        break;
                    }
                }
                tok = strtok(NULL, " \t\r\n");
            }
            if (*targetWordCount >= 30000) break;
        }
    }

    fclose(f);
    *targetFileLoaded = true;
    printf("[✓] Loaded %d tokens from File %d.\n", *targetWordCount, fileNumber);
}

// ====== Advanced Text Analysis Functions ======

// Load stopwords list
int load_stopwords(char stopwords[][MAX_WORD_LENGTH]) {
    FILE* file = fopen("stopwords.txt", "r");
    if (file == NULL) {
        printf("Error: Cannot open stopwords.txt\n");
        return 0;
    }

    int count = 0;
    char line[MAX_WORD_LENGTH];

    while (fgets(line, sizeof(line), file) && count < MAX_STOPWORDS) {
        // Remove newline character
        line[strcspn(line, "\n")] = '\0';
        // Convert to lowercase
        for (int i = 0; line[i]; i++) {
            line[i] = tolower(line[i]);
        }
        if (strlen(line) > 0) {
            strcpy(stopwords[count], line);
            count++;
        }
    }

    fclose(file);
    printf("Successfully loaded %d stopwords\n", count);
    return count;
}

// Check if a word is a stopword
int is_stopword(char* word, char stopwords[][MAX_WORD_LENGTH], int stop_count) {
    for (int i = 0; i < stop_count; i++) {
        if (strcmp(word, stopwords[i]) == 0) {
            return 1;
        }
    }
    return 0;
}

// Tokenize and clean text - SAFER VERSION
int tokenize_and_clean(char* text, struct WordInfo words[], char stopwords[][MAX_WORD_LENGTH], int stop_count, int* total_chars) {
    int word_count = 0;
    char* token;
    const char* delimiters = " .,!?;:\"\'()[]{}@#\n\t\r";

    printf("Text length: %lu characters\n", strlen(text));

    // Create a copy of text with size limit
    char* text_copy = malloc(strlen(text) + 1);
    if (text_copy == NULL) {
        printf("Error: Memory allocation failed\n");
        return 0;
    }
    strcpy(text_copy, text);

    token = strtok(text_copy, delimiters);
    int processed_tokens = 0;

    while (token != NULL && word_count < MAX_WORDS && processed_tokens < 50000) {
        processed_tokens++;

        // Clean word: convert to lowercase
        char clean_word[MAX_WORD_LENGTH];
        if (strlen(token) >= MAX_WORD_LENGTH) {
            // Skip words that are too long
            token = strtok(NULL, delimiters);
            continue;
        }

        strcpy(clean_word, token);
        for (int i = 0; clean_word[i]; i++) {
            clean_word[i] = tolower(clean_word[i]);
        }

        // Only keep words containing letters
        int has_letters = 0;
        for (int i = 0; clean_word[i]; i++) {
            if (isalpha(clean_word[i])) {
                has_letters = 1;
                break;
            }
        }

        if (has_letters && strlen(clean_word) > 0 && !is_stopword(clean_word, stopwords, stop_count)) {
            // Check if word already exists
            int found = 0;
            for (int i = 0; i < word_count; i++) {
                if (strcmp(words[i].word, clean_word) == 0) {
                    words[i].count++;
                    found = 1;
                    break;
                }
            }

            // If new word, add to array
            if (!found && word_count < MAX_WORDS) {
                strncpy(words[word_count].word, clean_word, MAX_WORD_LENGTH - 1);
                words[word_count].word[MAX_WORD_LENGTH - 1] = '\0'; // Ensure null termination
                words[word_count].count = 1;
                word_count++;
            }

            // Count characters
            *total_chars += strlen(clean_word);
        }

        token = strtok(NULL, delimiters);
    }

    free(text_copy); // Free allocated memory
    printf("Processed %d tokens, found %d unique words\n", processed_tokens, word_count);
    return word_count;
}

// Count number of sentences
int count_sentences(char* text) {
    int sentences = 0;
    for (int i = 0; text[i] && i < MAX_TEXT_LENGTH; i++) {
        if (text[i] == '.' || text[i] == '!' || text[i] == '?') {
            sentences++;
            // Skip consecutive punctuation
            while (text[i + 1] == '.' || text[i + 1] == '!' || text[i + 1] == '?') {
                i++;
            }
        }
    }
    return sentences > 0 ? sentences : 1;
}

// Generate statistics report
void generate_statistics(struct WordInfo words[], int word_count, int total_words, int total_chars, int sentences, int stopwords_removed) {
    printf("\n=== ADVANCED TEXT ANALYSIS REPORT ===\n");
    printf("Total words: %d\n", total_words);
    printf("Unique words: %d\n", word_count);
    printf("Number of sentences: %d\n", sentences);

    if (sentences > 0) {
        printf("Average sentence length: %.1f words\n", (float)total_words / sentences);
    }

    printf("Total characters: %d\n", total_chars);

    if (total_words > 0) {
        printf("Average word length: %.1f characters\n", (float)total_chars / total_words);
    }

    printf("Stopwords filtered: %d\n", stopwords_removed);

    if (total_words > 0) {
        float lexical_diversity = (float)word_count / total_words;
        printf("Lexical diversity: %.3f\n", lexical_diversity);
    }
}

// Bubble sort by frequency
void sort_by_frequency(struct WordInfo words[], int count) {
    for (int i = 0; i < count - 1; i++) {
        for (int j = 0; j < count - i - 1; j++) {
            if (words[j].count < words[j + 1].count) {
                // Swap
                struct WordInfo temp = words[j];
                words[j] = words[j + 1];
                words[j + 1] = temp;
            }
        }
    }
}

// Display top N frequent words
void show_top_words(struct WordInfo words[], int word_count, int n) {
    if (n > word_count) {
        n = word_count;
    }

    printf("\nTop %d most frequent words:\n", n);
    for (int i = 0; i < n; i++) {
        printf("%d. %s (%d occurrences)\n", i + 1, words[i].word, words[i].count);
    }
}

// Advanced Text Analysis function
void advancedTextAnalysis() {
    printf("\n=== Advanced Text Analysis ===\n");
    printf("This feature performs comprehensive text analysis with stopword filtering.\n");

    // Ask for file path
    char filePath[256];
    printf("Enter file path for analysis: ");
    if (!read_line(filePath, sizeof(filePath))) {
        handleError("Failed to read file path");
        return;
    }

    // Load stopwords
    char stopwords[MAX_STOPWORDS][MAX_WORD_LENGTH];
    int stop_count = load_stopwords(stopwords);
    if (stop_count == 0) {
        printf("Cannot continue: Stopwords file failed to load\n");
        return;
    }

    // Open input file
    FILE* input_file = open_file_read(filePath);
    if (input_file == NULL) {
        printf("ERROR: Cannot open file: %s\n", filePath);
        return;
    }

    // Read file content with size limit
    char text[MAX_TEXT_LENGTH] = "";
    char line[1000];
    int total_words_raw = 0;
    int stopwords_removed = 0;
    int total_chars = 0;

    printf("Reading file content...\n");
    while (fgets(line, sizeof(line), input_file) && strlen(text) < MAX_TEXT_LENGTH - 1000) {
        strcat(text, line);

        // Count raw words
        char line_copy[1000];
        strcpy(line_copy, line);
        char* token = strtok(line_copy, " .,!?;:\"\'()[]{}@#\n\t\r");
        while (token) {
            total_words_raw++;
            token = strtok(NULL, " .,!?;:\"\'()[]{}@#\n\t\r");
        }
    }
    fclose(input_file);

    printf("File reading completed. Total raw words: %d\n", total_words_raw);
    printf("Text buffer used: %lu/%d characters\n", strlen(text), MAX_TEXT_LENGTH);

    if (strlen(text) == 0) {
        printf("ERROR: No content read from file\n");
        return;
    }

    // Process text
    struct WordInfo* words = malloc(MAX_WORDS * sizeof(struct WordInfo));
    if (words == NULL) {
        printf("Error: Could not allocate memory for words array\n");
        return;
    }

    printf("Starting text processing...\n");
    int word_count = tokenize_and_clean(text, words, stopwords, stop_count, &total_chars);

    // Calculate total words after processing
    int total_words_clean = 0;
    for (int i = 0; i < word_count; i++) {
        total_words_clean += words[i].count;
    }

    stopwords_removed = total_words_raw - total_words_clean;

    // Count sentences
    int sentences = count_sentences(text);

    // Generate statistics report
    generate_statistics(words, word_count, total_words_clean, total_chars, sentences, stopwords_removed);

    // Sort and show frequent words
    sort_by_frequency(words, word_count);
    show_top_words(words, word_count, 15);

    // Free allocated memory
    free(words);

    printf("\nAdvanced analysis completed successfully!\n");
}

// ====== CSV and File Processing Functions ======

// Check if file is CSV format
bool isCSVFile(const char* filename) {
    const char* dot = strrchr(filename, '.');
    if (dot != NULL) {
        return (strcmp(dot, ".csv") == 0 || strcmp(dot, ".CSV") == 0);
    }
    return false;
}

// Process CSV file: convert columns to text
void processCSVFile(FILE* f, char (*targetWords)[50], int* targetWordCount) {
    char line[4096];
    int columnCount = 0;
    char* columns[100]; // Assume max 100 columns

    printf("[i] Processing CSV file...\n");

    while (fgets(line, sizeof(line), f) && *targetWordCount < 30000) {
        // Remove line endings
        line[strcspn(line, "\n\r")] = '\0';

        // CSV parsing: simple comma separation
        columnCount = 0;
        char* token = strtok(line, ",");
        while (token != NULL && columnCount < 100) {
            columns[columnCount++] = token;
            token = strtok(NULL, ",");
        }

        // Process each column
        for (int i = 0; i < columnCount && *targetWordCount < 30000; i++) {
            // Copy column content to temporary buffer
            char columnText[256];
            strncpy(columnText, columns[i], sizeof(columnText) - 1);
            columnText[sizeof(columnText) - 1] = '\0';

            // Normalize text
            normalize_line(columnText);

            // Tokenize
            char* word = strtok(columnText, " \t\r\n");
            while (word != NULL && *targetWordCount < 30000) {
                size_t wordLen = strlen(word);
                if (wordLen > 0 && wordLen < 50) {
                    strcpy(targetWords[*targetWordCount], word);
                    (*targetWordCount)++;
                }
                word = strtok(NULL, " \t\r\n");
            }
        }
    }
    printf("[✓] Processed %d columns from CSV\n", columnCount);
}

// ====== File Menu Handling Function ======
void handleFileMenu() {
    for (;;) {
        printf("\nFile Management Sub-Menu:\n");
        printf("1.1 - Load File 1\n");
        printf("1.2 - Load File 2\n");
        printf("1.3 - Show file history of File 1\n");
        printf("1.4 - Show file history of File 2\n");
        printf("1.5 - Exit to main menu\n");
        printf("Enter sub-choice (1.1 to 1.5): ");

        char subChoice[10];
        if (!read_line(subChoice, sizeof(subChoice))) {
            printf("Failed to read sub-choice.\n");
            continue;
        }

        if (strcmp(subChoice, "1.1") == 0) {
            printf("Enter file path for File 1 (supports .txt and .csv):\n> ");
            if (!read_line(inputFilePath1, sizeof(inputFilePath1))) {
                handleError("Failed to read file path");
                continue;
            }
            loadTextFile(1);
        }
        else if (strcmp(subChoice, "1.2") == 0) {
            printf("Enter file path for File 2 (supports .txt and .csv):\n> ");
            if (!read_line(inputFilePath2, sizeof(inputFilePath2))) {
                handleError("Failed to read file path");
                continue;
            }
            loadTextFile(2);
        }
        else if (strcmp(subChoice, "1.3") == 0) {
            showFileHistory(1);
        }
        else if (strcmp(subChoice, "1.4") == 0) {
            showFileHistory(2);
        }
        else if (strcmp(subChoice, "1.5") == 0) {
            break;  // Exit to main menu
        }
        else {
            printf("Invalid sub-choice. Please enter 1.1 to 1.5.\n");
        }
    }
}

// ====== Show File History Function (Shows CSV to TXT conversion) ======
void showFileHistory(int fileNumber) {
    char* filePath = (fileNumber == 1) ? inputFilePath1 : inputFilePath2;
    int wordCount = (fileNumber == 1) ? wordCount1 : wordCount2;
    bool fileLoaded = (fileNumber == 1) ? file1Loaded : file2Loaded;

    printf("\nFile %d History\n", fileNumber);
    printf("==================\n");
    if (!fileLoaded) {
        printf("No file loaded for File %d.\n", fileNumber);
        return;
    }

    // Remove quotes from file path for proper CSV detection
    char cleanPath[256];
    strncpy(cleanPath, filePath, sizeof(cleanPath) - 1);
    cleanPath[sizeof(cleanPath) - 1] = '\0';
    strip_quotes(cleanPath);

    // Determine original file type and processed file type
    bool isCSV = isCSVFile(cleanPath);
    const char* originalType = isCSV ? "CSV" : "Text";
    const char* processedType = "Text"; // CSV files are always converted to text format

    // Create converted file path (replace .csv with .txt if it's a CSV file)
    char convertedPath[256];
    strncpy(convertedPath, cleanPath, sizeof(convertedPath) - 1);
    convertedPath[sizeof(convertedPath) - 1] = '\0';

    if (isCSV) {
        // Replace .csv or .CSV extension with .txt
        char* dot = strrchr(convertedPath, '.');
        if (dot != NULL) {
            strcpy(dot, ".txt");
        }

        printf("Original file path: %s\n", cleanPath);
        printf("Converted file path: %s\n", convertedPath);
        printf("Original file type: %s\n", originalType);
        printf("Processed as: %s\n", processedType);
        printf("Note: CSV file converted to text format (columns extracted as words)\n");
    }
    else {
        // For text files, just show the original path
        printf("File path: %s\n", cleanPath);
        printf("File type: %s\n", originalType);
    }

    printf("Total words loaded: %d\n", wordCount);

    // Show first 10 words as sample
    char (*words)[50] = (fileNumber == 1) ? words1 : words2;
    int sampleCount = (wordCount < 10) ? wordCount : 10;
    printf("Sample words (%d): ", sampleCount);
    for (int i = 0; i < sampleCount; i++) {
        printf("%s", words[i]);
        if (i < sampleCount - 1) printf(", ");
    }
    printf("\n");
}

// ====== Updated: Display General Statistics ======
void displayGeneralStatistics() {
    if (!file1Loaded && !file2Loaded) {
        printf("[!] No text files loaded. Use menu 1 first.\n");
        return;
    }

    printf("General statistics:\n");
    printf("===================\n");

    if (file1Loaded) {
        int unique1 = 0;
        for (int i = 0; i < wordCount1; i++) {
            int seen = 0;
            for (int j = 0; j < i; j++) {
                if (strcmp(words1[i], words1[j]) == 0) { seen = 1; break; }
            }
            if (!seen) unique1++;
        }
        printf("File 1 (%s):\n", isCSVFile(inputFilePath1) ? "CSV" : "Text");
        printf("  Total words (tokens): %d\n", wordCount1);
        printf("  Unique words        : %d\n", unique1);
    }

    if (file2Loaded) {
        int unique2 = 0;
        for (int i = 0; i < wordCount2; i++) {
            int seen = 0;
            for (int j = 0; j < i; j++) {
                if (strcmp(words2[i], words2[j]) == 0) { seen = 1; break; }
            }
            if (!seen) unique2++;
        }
        printf("File 2 (%s):\n", isCSVFile(inputFilePath2) ? "CSV" : "Text");
        printf("  Total words (tokens): %d\n", wordCount2);
        printf("  Unique words        : %d\n", unique2);
    }
}

// ====== Updated: Sort and Display Top N Words ======
void sortAndDisplayTopNWords() {
    if (!file1Loaded && !file2Loaded) {
        printf("[!] No text files loaded. Use menu 1 first.\n");
        return;
    }

    int fileChoice;
    printf("Choose file to analyze:\n");
    if (file1Loaded) printf("1 - File 1 (%s)\n", isCSVFile(inputFilePath1) ? "CSV" : "Text");
    if (file2Loaded) printf("2 - File 2 (%s)\n", isCSVFile(inputFilePath2) ? "CSV" : "Text");
    printf("Enter choice: ");

    if (scanf("%d", &fileChoice) != 1) {
        int c; while ((c = getchar()) != '\n' && c != EOF);
        printf("Invalid number.\n");
        return;
    }
    int c; while ((c = getchar()) != '\n' && c != EOF);

    char (*words)[50];
    int wordCount;
    char* filePath;

    if (fileChoice == 1 && file1Loaded) {
        words = words1;
        wordCount = wordCount1;
        filePath = inputFilePath1;
    }
    else if (fileChoice == 2 && file2Loaded) {
        words = words2;
        wordCount = wordCount2;
        filePath = inputFilePath2;
    }
    else {
        printf("Invalid file choice or file not loaded.\n");
        return;
    }

    int n;
    printf("Enter N (a number) to get the top words: ");
    if (scanf("%d", &n) != 1) {
        int c; while ((c = getchar()) != '\n' && c != EOF);
        printf("Invalid number.\n");
        return;
    }
    int d; while ((d = getchar()) != '\n' && d != EOF);

    // Build unique word and frequency table
    char uniq[5000][50];
    int  freq[5000];
    int  ucnt = 0;

    for (int i = 0; i < wordCount; i++) {
        int k = -1;
        for (int j = 0; j < ucnt; j++) {
            if (strcmp(words[i], uniq[j]) == 0) { k = j; break; }
        }
        if (k == -1) {
            strcpy(uniq[ucnt], words[i]);
            freq[ucnt] = 1;
            ucnt++;
        }
        else {
            freq[k]++;
        }
    }

    // Simple sort: frequency descending, same frequency alphabetical
    for (int i = 0; i < ucnt - 1; i++) {
        for (int j = i + 1; j < ucnt; j++) {
            if (freq[j] > freq[i] || (freq[j] == freq[i] && strcmp(uniq[j], uniq[i]) < 0)) {
                int tf = freq[i]; freq[i] = freq[j]; freq[j] = tf;
                char tmp[50]; strcpy(tmp, uniq[i]); strcpy(uniq[i], uniq[j]); strcpy(uniq[j], tmp);
            }
        }
    }

    if (n > ucnt) n = ucnt;
    printf("Top %d words from File %d (%s):\n", n, fileChoice, isCSVFile(filePath) ? "CSV" : "Text");
    for (int i = 0; i < n; i++) {
        printf("%2d. %-20s %d\n", i + 1, uniq[i], freq[i]);
    }
}

// ====== Other Functions Remain Unchanged ======
void displayToxicWordAnalysis() {
    if (!file1Loaded && !file2Loaded) {
        printf("[!] No text loaded. Use menu 1 first.\n");
        return;
    }
    printf("[i] Toxic analysis is not implemented yet. (Next stage)\n");
    printf("Toxic words detected: %d (placeholder)\n", toxicCount);
}

void saveResultsToFile() {
    if (!file1Loaded && !file2Loaded) {
        printf("[!] No text loaded. Use menu 1 first.\n");
        return;
    }
    // Remove quotes before opening file
    char path[512];
    strncpy(path, outputFilePath, sizeof(path) - 1);
    path[sizeof(path) - 1] = '\0';
    strip_quotes(path);

#ifdef _WIN32
    FILE* f = fopen_u8(path, "w");
#else
    FILE* f = fopen(path, "w");
#endif
    if (!f) {
        fprintf(stderr, "[!] open failed for: %s\n", path);
        perror("reason: ");
        handleError("Cannot open output file");
        return;
    }

    // Simplified processing: only save first file's data
    char (*words)[50] = words1;
    int wordCount = wordCount1;

    // Calculate unique words and frequencies
    char uniq[1000][50];
    int  freq[5000];
    int  ucnt = 0;
    for (int i = 0; i < wordCount; i++) {
        int k = -1;
        for (int j = 0; j < ucnt; j++) {
            if (strcmp(words[i], uniq[j]) == 0) { k = j; break; }
        }
        if (k == -1) { strcpy(uniq[ucnt], words[i]); freq[ucnt] = 1; ucnt++; }
        else { freq[k]++; }
    }
    for (int i = 0; i < ucnt - 1; i++) {
        for (int j = i + 1; j < ucnt; j++) {
            if (freq[j] > freq[i] || (freq[j] == freq[i] && strcmp(uniq[j], uniq[i]) < 0)) {
                int tf = freq[i]; freq[i] = freq[j]; freq[j] = tf;
                char tmp[50]; strcpy(tmp, uniq[i]); strcpy(uniq[i], uniq[j]); strcpy(uniq[j], tmp);
            }
        }
    }

    fprintf(f, "Text Analysis Report\n");
    fprintf(f, "====================\n");
    fprintf(f, "Total tokens: %d\n", wordCount);
    fprintf(f, "Unique words: %d\n\n", ucnt);
    int topn = (ucnt < 10) ? ucnt : 10;
    fprintf(f, "Top %d words:\n", topn);
    for (int i = 0; i < topn; i++) {
        fprintf(f, "%2d. %-20s %d\n", i + 1, uniq[i], freq[i]);
    }
    fclose(f);
    printf("[✓] Saved report to %s\n", path);
}

void loadDictionaries() {
    printf("Dictionaries loaded (placeholder).\n");
}

void handleError(const char* message) {
    printf("Error: %s\n", message);
}

void translateTextToEnglish() {
    char sentence[256];
    printf("Enter a non-English sentence to translate: ");
    if (!read_line(sentence, sizeof(sentence))) {
        printf("Failed to read sentence.\n"); return;
    }
    printf("Translating...\nTranslated text: %s\n", sentence);
}

void displayNegativityScale() {
    printf("Negativity scale analysis (placeholder)\n");
}

void displayWordOccurrence() {
    printf("Word occurrence analysis (placeholder)\n");
}
