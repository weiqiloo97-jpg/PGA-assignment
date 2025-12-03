#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <ctype.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>

#define MAX_WORDS 3000000
#define MAX_WORD_LENGTH 50
#define MAX_STOPWORDS 500
#define MAX_TEXT_LENGTH 2000000
#define MAX_VARIANTS 300
#define MAX_TOXIC_WORDS 1000
#define MAX_PHRASES 500
#define MAX_TEXT_LENGTH_ADV 50000

#define ISALPHA(c) isalpha((unsigned char)(c))
#define TOLOWER(c) tolower((unsigned char)(c))

// ====== STAGE 2 DATA STRUCTURES ======
// Stores a unique word and its frequency count
struct WordInfo {
    char word[MAX_WORD_LENGTH];
    int count;
};

// Maps a non-standard word variant to a normalised base form
struct VariantMap {
    char variant[MAX_WORD_LENGTH];
    char standard[MAX_WORD_LENGTH];
};
// ========== END OF STAGE 2 STRUCTURES ==========

// ========== STAGE 3 DATA STRUCTURES ==========
struct ToxicWord {
    char word[MAX_WORD_LENGTH];
    int severity; // 1-5 scale
    int frequency;
};

struct ToxicPhrase {
    char phrase[MAX_WORD_LENGTH * 3];
    int severity; // 1-5 scale
    int frequency;
    int ngram_len; // 2 = bigram, 3 = trigram
};
// ========== END OF STAGE 3 STRUCTURES ==========

// ===== MASTER ANALYSIS DATA STRUCT =====
struct AnalysisData {
    char* text;
    struct WordInfo* words;
    int word_count;
    int total_words_filtered;
    int total_chars;
    int sentences;
    int stopwords_removed;
    int total_words_original;
    char stopwords[MAX_STOPWORDS][MAX_WORD_LENGTH];
    int stop_count;
    char** filtered_word_list;
    int filtered_word_count;
    struct VariantMap variant_mappings[MAX_VARIANTS];
    int variant_count;
    bool variant_processing_enabled;
    char** original_word_list;
    int original_word_count;
    bool text_filtered;

    // ===== STAGE 3 TOXICITY FIELDS =====
    struct ToxicWord toxic_words_list[MAX_TOXIC_WORDS];
    struct ToxicPhrase toxic_phrases_list[MAX_PHRASES];
    int toxic_words_count;
    int toxic_phrases_count;
    int total_toxic_occurrences;
    int severity_count[6]; // 1-5 for severity levels
    float toxicity_density;
    int bigram_toxic_occurrences;
    int trigram_toxic_occurrences;
    // ===== END STAGE 3 =====
};

// ===== SORTING SUPPORT STRUCTS =====
typedef enum { KEY_FREQ_DESC, KEY_ALPHA } SortKey;  //Sorting key: frequency descending or alphabetically
typedef enum { ALG_BUBBLE, ALG_QUICK, ALG_MERGE } SortAlg;  //Sorting algorithm selector
typedef struct {
    char word[50];
    int  count;
} Pair;
typedef struct {
    long long comps;   // Comparison count
    long long moves;   // Swap
    double    ms;      // Elapsed time in ms
} SortStats;

// ===== GLOBAL STATE VARIABLES =====
struct AnalysisData analysis_data;
//Currently active file name
char current_filename[256] = "";
char current_manual_filtered_filename[256] = "";
bool user_manually_saved_current_session = false;
static int g_toxic_loaded = 0;
static int g_toxic_count = 0;
static char g_toxic[500][MAX_WORD_LENGTH];

// Multi-file processing support
char inputFilePath1[256];  
char inputFilePath2[256];  
char outputFilePath[256];

char words1[MAX_WORDS][50];     // Token list for File 1
char words2[MAX_WORDS][50];     // Token list for File 2

int  wordCount1 = 0;       // Word count for File 1
int  wordCount2 = 0;       // Word count for File 2
int  toxicCount = 0;       // Reserved for Stage 3 expansion

bool file1Loaded = false;  // Whether File 1 has been loaded
bool file2Loaded = false;  // Whether File 2 has been loaded

// Global sort configuration defaults
static SortStats g_stats;
static SortKey g_key = KEY_FREQ_DESC;   
static SortAlg g_alg = ALG_BUBBLE;       
static int     g_topN = 10;           
static int g_use_secondary_tiebreak = 1; 
static int g_use_file = 1; // 1=File1, 2=File2

// Reset sorting statistics
static void stats_reset(void) { g_stats.comps = 0; g_stats.moves = 0; g_stats.ms = 0.0; }

// Compare two Pair objects using the selected key
static int cmp_pairs(const Pair* a, const Pair* b, SortKey key);
// Wrapper comparator that counts comparisons for statistics
static inline int cmp_with_stats(const Pair* a, const Pair* b, SortKey key) {
    g_stats.comps++;
    return cmp_pairs(a, b, key);
}

// Swap two Pair objects, tracking move operations
static inline void swap_pair(Pair* x, Pair* y) {
    Pair t = *x; *x = *y; *y = t; g_stats.moves += 3; 
}

// Return current time in milliseconds
static inline double now_ms(void) {
    return 1000.0 * clock() / CLOCKS_PER_SEC;
}

// Select tokens from File 1 or File 2 depending on global state
static void pick_tokens(char (**out_words)[50], int* out_count);

// Remove leading/trailing whitespace from a string
static void trim_inplace(char* s) {
    size_t len = strlen(s);
    size_t start = 0;

    while (start < len && isspace((unsigned char)s[start])) {
        start++;
    }
    while (len > start && isspace((unsigned char)s[len - 1])) {
        len--;
    }

    if (start > 0 || len < strlen(s)) {
        memmove(s, s + start, len - start);
    }
    s[len - start] = '\0';
}

// Convert a string to lowercase in place
static void tolower_inplace(char* s) {
    for (size_t i = 0; s[i]; ++i) {
        s[i] = (char)tolower((unsigned char)s[i]);
    }
}

// ====== FUNCTION DECLARATIONS ======
char get_menu_option(const char* valid_options, const char* prompt);

// ========== STAGE 2 FUNCTION DECLARATIONS ==========
int read_line(char* buf, size_t cap);
int load_stopwords(char stopwords[][MAX_WORD_LENGTH]);
int is_stopword(char* word, char stopwords[][MAX_WORD_LENGTH], int stop_count);
// Delimiters used for tokenisation 
static const char* DELIMS = " \t\r\n.,!?;:\"()[]{}@#<>/\\|*_~^`=+-&$%";
void process_text_file(const char* filename);
void cleanup_analysis_data();
void display_advanced_analysis_menu();
void word_analysis();
void save_filtered_word_list_auto(const char* filename);
void save_filtered_word_list();
void sort_by_frequency(struct WordInfo words[], int count);
void init_basic_variants();
int load_variant_mappings(const char* filename);
char* normalise_variant(char* word);
void toggle_variant_processing();
void reprocess_with_variants();
void add_token_to_analysis(const char* tok, int* removed_by_stopwords);
// ========== END OF STAGE 2 FUNCTION DECLARATIONS ==========

// ========== STAGE 3 FUNCTION DECLARATIONS ==========
void load_toxic_data(const char* filename);
void sync_toxic_systems(void);
int is_toxic_word(const char* word);
int get_toxic_severity(const char* word);
void detect_toxic_content(const char* word);
void detect_toxic_phrases();
void run_toxic_analysis();
void reset_toxic_counts();
bool file_exists(const char* filename);
int string_case_insensitive_compare(const char* s1, const char* s2);

// Stage 3 menu functions
void display_toxic_menu();
void toxic_analysis();
void dictionary_management();
void save_toxic_dictionary(const char* filename);
void view_all_toxic_words();

// Stage 3 utility functions
void calculate_toxicity_density();
void add_custom_toxic_word();
void remove_toxic_word();
int phrase_contains_toxic_words(const char* phrase, char found_words[][MAX_WORD_LENGTH], int max_found, int* max_severity);
void add_custom_toxic_phrase(const char* phrase);
// ========== END OF STAGE 3 FUNCTION DECLARATIONS ==========

// -------- UPDATED GENERAL FUNCTION DECLARATIONS --------
void loadTextFile(int fileNumber);
void saveResultsToFile();
void handleError(const char* message);
void showFileHistory(int fileNumber);
void handleFileMenu();
bool isCSVFile(const char* filename);
void processCSVFile(FILE* f, char (*targetWords)[50], int* targetWordCount);
bool isFileCorrupted(const char* filePath, const char* fileContent, size_t contentSize);

// ====== Stage 4 FUNCTION DECLARATIONS ======
static int  build_pairs_from_tokens(char (*words)[50], int wordCount, Pair out[], int maxOut);
static void sort_pairs(Pair a[], int n, SortKey key, SortAlg alg);
static int  load_toxicwords(void);
void        menu_sort_and_report(void);
static void merge_sort_pairs(Pair a[], int l, int r, SortKey key);
void sort_and_show_topN_all(SortKey key, SortAlg alg, int topN);
void sort_and_show_topN_toxic(SortAlg alg, int topN);
void compare_algorithms_topN(int topN);
void show_extra_summary(void);
void list_alpha_all(void);
// ========== END OF STAGE 4 FUNCTION DECLARATIONS ==========

// ===== 0. General Utilities ======

// Safely read a full line from stdin, stripping newline characters 
int read_line(char* buf, size_t cap) {
    if (!fgets(buf, (int)cap, stdin)) return 0;
    size_t n = strlen(buf);
    while (n && (buf[n - 1] == '\n' || buf[n - 1] == '\r'))
        buf[--n] = '\0';
    return 1;
}

// Get a single-character menu option from the user and validate it
char get_menu_option(const char* valid_options, const char* prompt) {
    char buffer[100];
    char option;

    while (1) {
        printf("%s", prompt);

        if (!read_line(buffer, sizeof(buffer))) {
            printf("Input error. Please try again.\n");
            continue;
        }

        // Check for empty input
        if (strlen(buffer) == 0) {
            printf("Please enter a option.\n");
            continue;
        }

        // Require exactly one character
        if (strlen(buffer) != 1) {
            printf("Error: Please enter exactly one character.\n");
            continue;
        }

        option = buffer[0];

        // Check if option is within the allowed set
        if (strchr(valid_options, option) != NULL) {
            return option;
        }
        else {
            if (strlen(valid_options) == 1) {
                printf("Invalid option. Please select option %c.\n", valid_options[0]);
            }
            else {
                printf("Invalid option. Please select from options %c to %c.\n",
                    valid_options[0], valid_options[strlen(valid_options) - 1]);
            }
        }
    }
}

// Print a generic error message to the console.
void handleError(const char* message) {
    printf("Error: %s\n", message);
}

// Normalise a file path: trim spaces and surrounding quotes
void clean_path(char* s) {
    size_t len = strlen(s);

    // Trim trailing whitespace
    while (len > 0 && isspace((unsigned char)s[len - 1])) {
        s[--len] = '\0';
    }

    // Remove surrounding double quotes, e.g. "filename"
    if (len >= 2 && s[0] == '"' && s[len - 1] == '"') {
        memmove(s, s + 1, len - 2);
        s[len - 2] = '\0';
    }
}

// Compare two strings ignoring letter case differences.
// Returns 0 if equal, otherwise the lexical difference.
int string_case_insensitive_compare(const char* s1, const char* s2) {
    while (*s1 && *s2) {
        if (tolower((unsigned char)*s1) != tolower((unsigned char)*s2)) {
            return tolower((unsigned char)*s1) - tolower((unsigned char)*s2);
        }
        s1++;
        s2++;
    }
    return tolower((unsigned char)*s1) - tolower((unsigned char)*s2);
}

// Check whether a file can be opened for reading.
// Returns true if the file exists.
bool file_exists(const char* filename) {
    FILE* file = fopen(filename, "r");
    if (file) {
        fclose(file);
        return true;
    }
    return false;
}

// Normalize a line to lowercase and replace non-alphanumeric characters with spaces.
static void normalize_line(char* s) {
    for (char* p = s; *p; ++p) {
        unsigned char c = (unsigned char)*p;
        if (isalnum(c)) *p = (char)tolower(c);
        else *p = ' ';
    }
}

// Remove surrounding double quotes from a path string if present.
static void strip_quotes(char* s) {
    size_t n = strlen(s);
    if (n >= 2 && s[0] == '"' && s[n - 1] == '"') {
        memmove(s, s + 1, n - 2);
        s[n - 2] = '\0';
    }
}

// Open a file using a UTF-8 path on Windows, converting to wide characters internally.
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

// Open a file in read mode with basic path cleaning and cross-platform support.
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
        fprintf(stderr, "\n[!] Failed to open file for: %s\n", tmp);
        perror("reason");
    }
    return f;
}

// Allocate a MAX_WORDS x MAX_WORD_LENGTH string table
static int alloc_string_table(char*** table) {
    if (*table != NULL) {
        return 1;
    }

    *table = (char**)calloc(MAX_WORDS, sizeof(char*));
    if (!*table) {
        printf("Error: Memory allocation failed (table)\n");
        return 0;
    }

    for (int i = 0; i < MAX_WORDS; i++) {
        (*table)[i] = (char*)malloc(MAX_WORD_LENGTH);
        if (!(*table)[i]) {
            printf("Error: Memory allocation failed (row %d)\n", i);
            // Free already allocated rows on failure
            for (int j = 0; j < i; j++) {
                free((*table)[j]);
            }
            free(*table);
            *table = NULL;
            return 0;
        }
    }
    return 1;
}

// Helper: close file and clean analysis data on failure
static void fail_and_cleanup(FILE* f) {
    if (f) fclose(f);
    cleanup_analysis_data();
}


// ===== 1. Stage 1 - File Management, CSV and History =====

// Check whether a filename has a .csv extension (case-insensitive).
bool isCSVFile(const char* filename) {
    const char* dot = strrchr(filename, '.');
    if (dot != NULL) {
        return (strcmp(dot, ".csv") == 0 || strcmp(dot, ".CSV") == 0);
    }
    return false;
}

// Parse a CSV file, normalize each column, and tokenise into words.
void processCSVFile(FILE* f, char (*targetWords)[50], int* targetWordCount) {
    char line[4096];
    int columnCount = 0;
    char* columns[100]; // Assume at most 100 columns per row.

    printf("Processing your CSV file...\n");

    while (fgets(line, sizeof(line), f) && *targetWordCount < MAX_WORDS) {
        // Remove trailing newline characters.
        line[strcspn(line, "\n\r")] = '\0';

        // Simple CSV parsing: split the line by commas.
        columnCount = 0;
        char* token = strtok(line, ",");
        while (token != NULL && columnCount < 100) {
            columns[columnCount++] = token;
            token = strtok(NULL, ",");
        }

        // Process each column independently.
        for (int i = 0; i < columnCount && *targetWordCount < MAX_WORDS; i++) {
            // Copy the column text into a temporary buffer.
            char columnText[256];
            strncpy(columnText, columns[i], sizeof(columnText) - 1);
            columnText[sizeof(columnText) - 1] = '\0';

            // Normalise the text (e.g. lowercasing, stripping punctuation, etc.).
            normalize_line(columnText);

            // Tokenise the normalised text into individual words.
            char* word = strtok(columnText, " \t\r\n");
            while (word != NULL && *targetWordCount < MAX_WORDS) {
                size_t wordLen = strlen(word);
                if (wordLen > 0 && wordLen < 50) {
                    strcpy(targetWords[*targetWordCount], word);
                    (*targetWordCount)++;
                }
                word = strtok(NULL, " \t\r\n");
            }
        }
    }
    printf("Processed %d columns from CSV file\n", columnCount);
}

//file corruption detection - FIXED VERSION
bool isFileCorrupted(const char* filePath, const char* fileContent, size_t contentSize) {
    //prevent crash for empty file
    if (contentSize == 0) {
        printf("\n[X] ERROR: Empty file detected!\n");
        printf("\nRecovery Guide:\n");
        printf("1. The file '%s' is completely empty (0 bytes)\n", filePath);
        printf("2. Please select a file that contains actual text content\n");
        printf("3. Ensure the file has readable text before loading\n");
        printf("4. Try opening the file in a text editor to verify it has content\n");
        return true;
    }

    int totalChars = 0;
    int printableChars = 0;
    int controlChars = 0;
    int weirdChars = 0;
    int consecutiveWeird = 0;
    int maxConsecutiveWeird = 0;
    int wordLikeSequences = 0;
    int binaryMarkers = 0;

    //weird character detection? as in non-printable characters
    for (size_t i = 0; i < contentSize; i++) {
        unsigned char c = (unsigned char)fileContent[i];
        totalChars++;

        //normal characters: printable ASCII + common whitespace
        if ((c >= 32 && c <= 126) || c == '\t' || c == '\n' || c == '\r') {
            printableChars++;
            consecutiveWeird = 0;

            // Detect word-like sequences (letter sequences)
            if (isalpha(c)) {
                if (i == 0 || !isalpha((unsigned char)fileContent[i - 1])) {
                    size_t j = i;
                    int wordLen = 0;
                    while (j < contentSize && isalpha((unsigned char)fileContent[j])) {
                        wordLen++;
                        j++;
                    }
                    if (wordLen >= 2 && wordLen <= 25) {
                        wordLikeSequences++;
                    }
                }
            }
        }
        else {
            // This character is "weird" (non-printable, non-whitespace)
            weirdChars++;
            consecutiveWeird++;
            if (consecutiveWeird > maxConsecutiveWeird) {
                maxConsecutiveWeird = consecutiveWeird;
            }

            // Also count as control character for additional metrics
            controlChars++;
        }
    }

    // Calculate ratios
    double printableRatio = (double)printableChars / totalChars;
    double weirdRatio = (double)weirdChars / totalChars;
    double controlCharRatio = (double)controlChars / totalChars;
    double wordDensity = (double)wordLikeSequences / (contentSize / 100.0);

    printf("File analysis: %zu chars, %.1f%% printable, %.1f%% weird, word density: %.1f/100chars\n",
        contentSize, printableRatio * 100, weirdRatio * 100, wordDensity);

    bool likelyCorrupted = false;

    // MAIN CONDITION: Only corrupt if more than 50% of file is weird characters
    if (weirdRatio > 0.50) {
        likelyCorrupted = true;
    }
    else if (printableRatio < 0.10 && contentSize > 100) {
        likelyCorrupted = true;
    }
    else if (maxConsecutiveWeird > 100) {
        likelyCorrupted = true;
    }
    else if (wordDensity < 0.1 && contentSize > 500 && weirdRatio > 0.30) {
        likelyCorrupted = true;
    }

    if (likelyCorrupted) {
        printf("\n[X] ERROR: Corrupted file detected!\n");
        printf("\nRecovery Guide:\n");
        printf("1. This file contains too many nonsense characters (>50%%)\n");
        printf("2. Please select a valid text file with mostly readable content\n");
        printf("3. Ensure the file contains proper text.\n");
        return true;
    }
    printf("File is valid! Your file contains %.1f%% weird characters\n", weirdRatio * 100);
    return false;
}

//Updated: Load Text File Function (Supports CSV and corrupted file detection)
void loadTextFile(int fileNumber) {
    char* filePath = (fileNumber == 1) ? inputFilePath1 : inputFilePath2;
    char (*targetWords)[50] = (fileNumber == 1) ? words1 : words2;
    int* targetWordCount = (fileNumber == 1) ? &wordCount1 : &wordCount2;
    bool* targetFileLoaded = (fileNumber == 1) ? &file1Loaded : &file2Loaded;

    // open_file_read uses the path stored in filePath and attempts to open it.
    FILE* f = open_file_read(filePath);
    if (!f) {
        // Failed to open → mark this file as “not loaded”.
        *targetFileLoaded = false;
        *targetWordCount = 0;
        printf("Recovery Guide:\n");
        printf("1. Make sure the file name is correct\n");
        printf("2. Move the file to the same directory as this program\n");
        return;
    }

    // Important: once the file opens successfully, clean the global filePath (remove quotes and trailing problematic characters).
    clean_path(filePath);

    *targetWordCount = 0;

    // Use the cleaned path for file-type and corruption checks.
    char cleanPath[256];
    strncpy(cleanPath, filePath, sizeof(cleanPath) - 1);
    cleanPath[sizeof(cleanPath) - 1] = '\0';

    // ====== New: File content pre-scan and corruption detection ======
    printf("\nFile corruption detection ongoing.....\n");

    // Read entire file content for detection
    fseek(f, 0, SEEK_END);
    long fileSize = ftell(f);
    fseek(f, 0, SEEK_SET);

    // Check for empty file first
    if (fileSize <= 0) {
        printf("\n[X] ERROR: File is empty!\n");
        printf("Please load a file with text content.\n");
        fclose(f);
        *targetFileLoaded = false;
        return;
    }

    if (fileSize > 0) {
        char* fileContent = (char*)malloc(fileSize + 1);
        if (fileContent) {
            size_t bytesRead = fread(fileContent, 1, fileSize, f);
            fileContent[bytesRead] = '\0';

            // Detect if file is corrupted or empty
            if (isFileCorrupted(cleanPath, fileContent, bytesRead)) {
                free(fileContent);
                fclose(f);
                *targetFileLoaded = false;
                return; // Stop loading corrupted/empty file
            }

            free(fileContent);
        }
        // Reset file pointer to beginning
        fseek(f, 0, SEEK_SET);
    }

    // Check file type and process accordingly
    if (isCSVFile(cleanPath)) {
        printf("Detected CSV file format: now converting columns to text...\n");
        processCSVFile(f, targetWords, targetWordCount);
        printf("CSV file converted to text format!\n");
    }
    else {
        printf("Detected text file format: processing as text...\n");
        char line[4096];
        while (fgets(line, sizeof(line), f)) {
            normalize_line(line); // Lowercase + non-alphanumeric = space
            char* tok = strtok(line, " \t\r\n");
            while (tok) {
                size_t L = strlen(tok);
                if (L > 0 && L < sizeof(targetWords[0])) {
                    if (*targetWordCount < MAX_WORDS) {
                        strcpy(targetWords[*targetWordCount], tok);
                        (*targetWordCount)++;
                    }
                    else {
                        printf("[!] Warning: Reached maximum token limit.\n");
                        break;
                    }
                }
                tok = strtok(NULL, " \t\r\n");
            }
            if (*targetWordCount >= MAX_WORDS) break;
        }
    }
    fclose(f);

    if (*targetWordCount == 0) {
        printf("[!] Warning: File loaded but no valid words found\n");
        *targetFileLoaded = false;
    }
    else {
        *targetFileLoaded = true;
        printf("Loaded %d tokens from File %d.\n", *targetWordCount, fileNumber);
    }
}

//Handle the file management submenu : loading files and viewing file history.
void handleFileMenu() {
    for (;;) {
        printf("\n=== File Management Menu ===\n");
        printf("1. Load File 1 (Current: %s)\n", file1Loaded ? inputFilePath1 : "No file loaded");
        printf("2. Load File 2 (Current: %s)\n", file2Loaded ? inputFilePath2 : "No file loaded");
        printf("3. View file history of File 1\n");
        printf("4. View file history of File 2\n");
        printf("5. Exit to main menu\n");
        printf("Enter sub-choice (1 to 5): ");

        char subChoice[10];
        if (!read_line(subChoice, sizeof(subChoice))) {
            printf("Failed to read sub-choice.\n");
            continue;
        }

        // Load a file into slot 1
        if (strcmp(subChoice, "1") == 0) {
            printf("Enter file path for File 1 (supports .txt and .csv):\n> ");
            if (!read_line(inputFilePath1, sizeof(inputFilePath1))) {
                handleError("Failed to read file path");
                continue;
            }
            if (strlen(inputFilePath1) == 0) {
                printf("[X] Error. Please load a file!\n");
                continue;
            }
            clean_path(inputFilePath1);
            loadTextFile(1);
        }
        // Load a file into slot 2
        else if (strcmp(subChoice, "2") == 0) {
            printf("Enter file path for File 2 (supports .txt and .csv):\n> ");
            if (!read_line(inputFilePath2, sizeof(inputFilePath2))) {
                handleError("Failed to read file path");
                continue;
            }
            if (strlen(inputFilePath2) == 0) {
                printf("No filename provided.\n");
                continue;
            }
            clean_path(inputFilePath2);
            loadTextFile(2);
        }
        // Display history of File 1
        else if (strcmp(subChoice, "3") == 0) {
            showFileHistory(1);
        }
        // Display history of File 2
        else if (strcmp(subChoice, "4") == 0) {
            showFileHistory(2);
        }
        // Return to main menu
        else if (strcmp(subChoice, "5") == 0) {
            printf("Returning to main menu...\n");
            break;
        }
        else {
            printf("Invalid sub-choice. Please enter 1 to 5\n");
        }
    }
}

// Show basic information and sample tokens for the selected file slot.
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

    // Identify original file type and indicate the processed format.
    bool isCSV = isCSVFile(cleanPath);
    const char* originalType = isCSV ? "CSV" : "txt";
    const char* processedType = "txt"; // CSV files are always converted to text format

    // Build an equivalent .txt path for CSV files.
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
    }
    else {
        // Show basic metadata for text files.
        printf("File path: %s\n", cleanPath);
        printf("File type: %s\n", originalType);
    }

    printf("Total words loaded: %d\n", wordCount);

    //  Display the first 10 words as a preview sample.
    char (*words)[50] = (fileNumber == 1) ? words1 : words2;
    int sampleCount = (wordCount < 10) ? wordCount : 10;
    printf("Sample words (%d): ", sampleCount);
    for (int i = 0; i < sampleCount; i++) {
        printf("%s", words[i]);
        if (i < sampleCount - 1) printf(", ");
    }
    printf("\n");
}

// Automatically select active source file based on loaded flags 
// Returns false if no file is loaded, true otherwise
bool auto_select_source_file(void) {
    if (!file1Loaded && !file2Loaded) {
        printf("[X] No files loaded yet. Please load at least one file in Stage 1 first.\n");
        return false;
    }
    else if (!file1Loaded && file2Loaded) {
        g_use_file = 2;
    }
    else if (file1Loaded && !file2Loaded) {
        g_use_file = 1;
    }
    // If both are loaded, keep existing g_use_file unchanged
    return true;
}

// Allow the user to switch between File 1 and File 2 as the active source
void prompt_change_source_file(void) {
    int s;
    bool changed = false;
    printf("Choose source: 1 = File 1  2 = File 2: ");
    if (scanf("%d", &s) != 1) {
        printf("Invalid input.\n");
    }
    else if (s == 1) {
        if (!file1Loaded) {
            printf("Error: File 1 is not loaded yet. Please load File 1 in Stage 1 first.\n");
        }
        else {
            g_use_file = 1;
            printf("Source set to File 1: %s\n", inputFilePath1);
        }
    }
    else if (s == 2) {
        if (!file2Loaded) {
            printf("Error: File 2 is not loaded yet. Please load File 2 in Stage 1 first.\n");
        }
        else {
            g_use_file = 2;
            printf("Source set to File 2: %s\n", inputFilePath2);
        }
    }
    else {
        printf("Invalid source. Keep current.\n");
    }

    // Flush any extra input to avoid breaking future scanf calls
    int c;
    while ((c = getchar()) != '\n' && c != EOF);
}

// Pick the currently active token buffer (File 1 or File 2) based on global flags.
static void pick_tokens(char (**out_words)[50], int* out_count) {
    if (g_use_file == 1 && file1Loaded) { *out_words = words1; *out_count = wordCount1; return; }
    if (g_use_file == 2 && file2Loaded) { *out_words = words2; *out_count = wordCount2; return; }
    // Auto mode: fall back to the default priority (prefer File 1 if available, otherwise File 2).
    if (file1Loaded) { *out_words = words1; *out_count = wordCount1; return; }
    if (file2Loaded) { *out_words = words2; *out_count = wordCount2; return; }
    *out_words = NULL; *out_count = 0;
}



// ====== 2. Stage 2 - Advanced Text Analysis (stopwords, variants, statistics) =====

// Load the stopword list from stopwords.txt into memory
int load_stopwords(char stopwords[][MAX_WORD_LENGTH]) {
    FILE* file = fopen("stopwords.txt", "r");
    if (file == NULL) {
        printf("Error: Cannot open stopwords.txt\n");
        printf("Please ensure stopwords.txt is in the same directory.\n");
        return 0;
    }

    int count = 0;
    char line[MAX_WORD_LENGTH];

    while (fgets(line, sizeof(line), file) && count < MAX_STOPWORDS) {
        line[strcspn(line, "\n")] = '\0';
        // Convert to lowercase for consistent matching
        for (int i = 0; line[i]; i++) {
            line[i] = (char)TOLOWER(line[i]);
        }
        if (strlen(line) > 0) {
            strcpy(stopwords[count], line);
            count++;
        }
    }

    fclose(file);
    printf("Loaded %d stopwords from stopwords.txt\n", count);
    return count;
}

// Check whether a word is a stopword
int is_stopword(char* word, char stopwords[][MAX_WORD_LENGTH], int stop_count) {
    for (int i = 0; i < stop_count; i++) {
        if (strcmp(word, stopwords[i]) == 0) {
            return 1;
        }
    }
    return 0;
}

// Initialise core variant mappings
void init_basic_variants() {
    // Keep only essential core mappings; others live in the external file
    const char* core_mappings[][2] = {
        {"u", "you"},
        {"ur", "your"},
        {"r", "are"},
        {"lol", "laughing out loud"},
        {"btw", "by the way"},
        {"omg", "oh my god"}
    };

    int num_core = sizeof(core_mappings) / sizeof(core_mappings[0]);
    for (int i = 0; i < num_core && analysis_data.variant_count < MAX_VARIANTS; i++) {
        strcpy(analysis_data.variant_mappings[analysis_data.variant_count].variant, core_mappings[i][0]);
        strcpy(analysis_data.variant_mappings[analysis_data.variant_count].standard, core_mappings[i][1]);
        analysis_data.variant_count++;
    }
    // Load additional mappings from file
    int loaded = load_variant_mappings("variant_mappings.txt");
    printf("Initialised %d core variants + loaded %d mappings from file (total %d, limit %d)\n",
        num_core, loaded, analysis_data.variant_count, MAX_VARIANTS);
}

// Load extra variant mappings from a key=value text file
int load_variant_mappings(const char* filename) {
    FILE* file = fopen(filename, "r");
    if (file == NULL) {
        return 0;
    }

    int loaded = 0;
    char line[256];

    while (fgets(line, sizeof(line), file) && analysis_data.variant_count < MAX_VARIANTS) {
        if (line[0] == '\n' || line[0] == '#' || line[0] == '\r') {
            continue;
        }

        line[strcspn(line, "\n")] = '\0';
        line[strcspn(line, "\r")] = '\0';

        // Skip empty lines or comments
        if (line[0] == '\0' || line[0] == '#')
            continue;

        char* separator = strchr(line, '=');
        if (separator == NULL) {
            continue;
        }

        *separator = '\0';
        char* variant = line;
        char* standard = separator + 1;

        // Trim leading spaces
        while (*variant == ' ') variant++;
        while (*standard == ' ') standard++;

        // Trim trailing spaces
        char* end = variant + strlen(variant) - 1;
        while (end > variant && *end == ' ') *end-- = '\0';

        end = standard + strlen(standard) - 1;
        while (end > standard && *end == ' ') *end-- = '\0';

        if (strlen(variant) > 0 && strlen(standard) > 0) {
            strncpy(analysis_data.variant_mappings[analysis_data.variant_count].variant,
                variant, MAX_WORD_LENGTH - 1);
            strncpy(analysis_data.variant_mappings[analysis_data.variant_count].standard,
                standard, MAX_WORD_LENGTH - 1);

            analysis_data.variant_mappings[analysis_data.variant_count].variant[MAX_WORD_LENGTH - 1] = '\0';
            analysis_data.variant_mappings[analysis_data.variant_count].standard[MAX_WORD_LENGTH - 1] = '\0';

            analysis_data.variant_count++;
            loaded++;
        }
    }

    fclose(file);
    return loaded;
}

// Normalise a word using the variant mapping table if enabled
char* normalise_variant(char* word) {
    if (!analysis_data.variant_processing_enabled) {
        return word;
    }

    for (int i = 0; i < analysis_data.variant_count; i++) {
        if (strcmp(word, analysis_data.variant_mappings[i].variant) == 0) {
            return analysis_data.variant_mappings[i].standard;
        }
    }
    return word;
}

//extra feature (last resort)
void toggle_variant_processing() {
    // First show current status
    printf("\nCurrent Text Normalisation: %s\n",
        analysis_data.variant_processing_enabled ? "ENABLED" : "DISABLED");

    // Show sample from real variant mappings
    printf("\n=== TEXT NORMALISATION EXAMPLES ===\n");
    printf("Below are sample normalisation mappings used by the system:\n\n");

    printf("+-----------------+-----------------------+\n");
    printf("| Input Form      | Normalised To         |\n");
    printf("+-----------------+-----------------------+\n");

    int limit = (analysis_data.variant_count < 10) ? analysis_data.variant_count : 10;

    for (int i = 0; i < limit; i++) {
        printf("| %-15s | %-21s |\n", analysis_data.variant_mappings[i].variant, analysis_data.variant_mappings[i].standard);
    }

    printf("+-----------------+-----------------------+\n");
    printf("(Showing %d of %d mappings loaded)\n", limit, analysis_data.variant_count);

    // Ask user if they want to toggle normalisation
    printf("\nDo you want to %s text normalisation? (y/n): ",
        analysis_data.variant_processing_enabled ? "DISABLE" : "ENABLE");

    char response = getchar();
    getchar(); // Clear newline

    if (response == 'y' || response == 'Y') {
        // Toggle the setting
        analysis_data.variant_processing_enabled = !analysis_data.variant_processing_enabled;

        printf("\nText Normalisation is now %s\n",
            analysis_data.variant_processing_enabled ? "ENABLED" : "DISABLED");

        // Re-process text if we have a file loaded
        if (analysis_data.text_filtered) {
            printf("\nRe-processing text with new normalisation setting...\n");
            int previous_word_count = analysis_data.total_words_filtered;
            int previous_unique_words = analysis_data.word_count;

            reprocess_with_variants();

            printf("Text re-processing completed!\n");

            // Show statistics changes
            int word_change = analysis_data.total_words_filtered - previous_word_count;
            int unique_change = analysis_data.word_count - previous_unique_words;

            printf("\nText statistics updated:\n");
            printf("  * Total words: %d -> %d (%+d)\n",
                previous_word_count, analysis_data.total_words_filtered, word_change);
            printf("  * Unique words: %d -> %d (%+d)\n",
                previous_unique_words, analysis_data.word_count, unique_change);

            if (word_change != 0) {
                printf("  * Change due to normalisation: %+d words\n", word_change);
            }
        }
    }
    else {
        printf("Text Normalisation setting unchanged.\n");
    }
}

// Add one token into the analysis pipeline
void add_token_to_analysis(const char* tok, int* removed_by_stopwords) {
    if (!tok || !*tok) return;

    // Check that token contains at least one alphabetic character
    int has_letters = 0;
    for (int j = 0; tok[j]; j++) {
        if (ISALPHA(tok[j])) { has_letters = 1; break; }
    }
    if (!has_letters) return;

    // Skip if token is a stopword
    if (is_stopword((char*)tok, analysis_data.stopwords, analysis_data.stop_count)) {
        (*removed_by_stopwords)++;
        return;
    }

    // Append token to filtered word list
    if (analysis_data.filtered_word_list != NULL &&
        analysis_data.filtered_word_count < MAX_WORDS) {
        strncpy(analysis_data.filtered_word_list[analysis_data.filtered_word_count],
            tok, MAX_WORD_LENGTH - 1);
        analysis_data.filtered_word_list[analysis_data.filtered_word_count][MAX_WORD_LENGTH - 1] = '\0';
    }
    analysis_data.filtered_word_count++;
    analysis_data.total_words_filtered++;
    analysis_data.total_chars += (int)strlen(tok);

    // Update frequency statistics for unique words
    if (analysis_data.words != NULL) {
        int found = 0;
        for (int k = 0; k < analysis_data.word_count; k++) {
            if (strcmp(analysis_data.words[k].word, tok) == 0) {
                analysis_data.words[k].count++;
                found = 1;
                break;
            }
        }
        if (!found && analysis_data.word_count < MAX_WORDS) {
            strcpy(analysis_data.words[analysis_data.word_count].word, tok);
            analysis_data.words[analysis_data.word_count].count = 1;
            analysis_data.word_count++;
        }
    }
}

// Dynamically reprocess text using current variant & stopword settings
void reprocess_with_variants() {
    if (analysis_data.original_word_list == NULL) return;

    // Reset counters for the new pass
    analysis_data.total_words_filtered = 0;
    analysis_data.total_chars = 0;
    analysis_data.word_count = 0;
    analysis_data.filtered_word_count = 0;

    if (analysis_data.words != NULL) {
        memset(analysis_data.words, 0, MAX_WORDS * sizeof(struct WordInfo));
    }

    int variants_normalised = 0;
    int removed_by_stopwords = 0;
    int considered_tokens = 0;

    for (int i = 0; i < analysis_data.original_word_count && analysis_data.filtered_word_count < MAX_WORDS; i++) {
        char current_word[MAX_WORD_LENGTH];
        strncpy(current_word, analysis_data.original_word_list[i], MAX_WORD_LENGTH - 1);
        current_word[MAX_WORD_LENGTH - 1] = '\0';
        if (!*current_word) continue;

        // Apply variant mapping if enabled
        char* normalised = normalise_variant(current_word);

        if (normalised != current_word) {
            variants_normalised++;

            // Phrase mapping: split into multiple tokens
            if (strchr(normalised, ' ') != NULL) {
                char phrase_buf[MAX_WORD_LENGTH * 4];
                strncpy(phrase_buf, normalised, sizeof(phrase_buf) - 1);
                phrase_buf[sizeof(phrase_buf) - 1] = '\0';

                char* part = strtok(phrase_buf, " ");
                while (part && analysis_data.filtered_word_count < MAX_WORDS) {
                    int has_letters = 0;
                    for (int j = 0; part[j]; j++) {
                        if (ISALPHA(part[j])) {
                            has_letters = 1;
                            break;
                        }
                    }

                    if (*part && has_letters) {
                        considered_tokens++;
                        add_token_to_analysis(part, &removed_by_stopwords);
                    }
                    part = strtok(NULL, " ");
                }
                continue;
            }

            // Single-word mapping
            strncpy(current_word, normalised, MAX_WORD_LENGTH - 1);
            current_word[MAX_WORD_LENGTH - 1] = '\0';
        }

        // Check that current_word contains letters
        int has_letters = 0;
        for (int j = 0; current_word[j]; j++) {
            if (ISALPHA(current_word[j])) {
                has_letters = 1;
                break;
            }
        }

        if (has_letters) {
            considered_tokens++;
            add_token_to_analysis(current_word, &removed_by_stopwords);
        }
    }

    analysis_data.stopwords_removed = considered_tokens - analysis_data.total_words_filtered;

    if (analysis_data.variant_processing_enabled && variants_normalised > 0) {
        printf("  - Text forms normalised: %d (abbreviations and Leet Speak)\n", variants_normalised);
    }
}

// Process and analyse a text file with stopwords & variants
void process_text_file(const char* filename) {
    // Copy filename into a local buffer, then clean the path
    char path_buf[256];
    strncpy(path_buf, filename, sizeof(path_buf) - 1);
    path_buf[sizeof(path_buf) - 1] = '\0';

    // Clean path: remove quotes and stray characters）
    clean_path(path_buf);

    printf("\nProcessing file: %s\n", path_buf);
    strncpy(current_filename, path_buf, sizeof(current_filename) - 1);
    current_filename[sizeof(current_filename) - 1] = '\0';

    // Clear previous analysis state
    cleanup_analysis_data();

    // Load stopwords before processing
    analysis_data.stop_count = load_stopwords(analysis_data.stopwords);
    if (analysis_data.stop_count == 0) {
        printf("Cannot continue without stopwords.\n");
        return;
    }

    // Open the file using cleaned path
    FILE* file = fopen(path_buf, "r");
    if (file == NULL) {
        printf("ERROR: Cannot open file: %s\n", path_buf);
        return;
    }

    // Allocate main text buffer
    analysis_data.text = (char*)malloc(MAX_TEXT_LENGTH);
    if (!analysis_data.text) {
        printf("Error: Memory allocation failed (text)\n");
        fail_and_cleanup(file);
        return;
    }
    memset(analysis_data.text, 0, MAX_TEXT_LENGTH);

    // Allocate tables for filtered and original words
    analysis_data.words = (struct WordInfo*)calloc(MAX_WORDS, sizeof(struct WordInfo));
    if (!analysis_data.words) {
        printf("Error: Memory allocation failed (words)\n");
        fail_and_cleanup(file);
        return;
    }

    if (!alloc_string_table(&analysis_data.filtered_word_list)) {
        fail_and_cleanup(file);
        return;
    }
    if (!alloc_string_table(&analysis_data.original_word_list)) {
        fail_and_cleanup(file);
        return;
    }

    size_t used = 0;
    analysis_data.total_words_original = 0;
    char line[1000];

    printf("Reading file content...\n");
    while (fgets(line, sizeof(line), file)) {
        size_t len = strlen(line);
        if (used + len + 1 >= MAX_TEXT_LENGTH) break;
        memcpy(analysis_data.text + used, line, len);
        used += len;
        analysis_data.text[used] = '\0';
    }
    fclose(file);

    // Replace non-ASCII bytes with spaces for safety
    for (size_t i = 0; analysis_data.text[i]; ++i) {
        unsigned char ch = (unsigned char)analysis_data.text[i];
        if (ch > 127) analysis_data.text[i] = ' ';
    }

    if (analysis_data.text[0] == '\0') {
        printf("ERROR: No content read from file\n");
        return;
    }

    // Tokenisation pass: collect original words
    printf("Starting text processing...\n");
    char* text_copy = (char*)malloc(strlen(analysis_data.text) + 1);
    if (!text_copy) {
        printf("Error: Memory allocation failed\n");
        return;
    }
    strcpy(text_copy, analysis_data.text);

    char* token = strtok(text_copy, DELIMS);
    analysis_data.total_chars = 0;
    analysis_data.word_count = 0;
    analysis_data.filtered_word_count = 0;
    analysis_data.total_words_filtered = 0;
    analysis_data.original_word_count = 0;

    // Collect all original tokens before variant / stopword filtering
    while (token && analysis_data.original_word_count < MAX_WORDS) {
        char clean_word[MAX_WORD_LENGTH];
        strncpy(clean_word, token, MAX_WORD_LENGTH - 1);
        clean_word[MAX_WORD_LENGTH - 1] = '\0';

        // Strip special prefixes like hashtags or mentions
        if (clean_word[0] == '#' || clean_word[0] == '@') {
            memmove(clean_word, clean_word + 1, strlen(clean_word));
        }

        // Lowercase & keep ASCII only
        for (int i = 0; clean_word[i]; i++) {
            clean_word[i] = (char)tolower((unsigned char)clean_word[i]);
        }

        // Save processed original token
        if (strlen(clean_word) > 0) {
            strncpy(analysis_data.original_word_list[analysis_data.original_word_count],
                clean_word, MAX_WORD_LENGTH - 1);
            analysis_data.original_word_list[analysis_data.original_word_count][MAX_WORD_LENGTH - 1] = '\0';
            analysis_data.original_word_count++;
            analysis_data.total_words_original++;
        }

        token = strtok(NULL, DELIMS);
    }

    free(text_copy);

    // Apply variant mappings and stopword filtering
    reprocess_with_variants();
    printf("File reading completed. Total words in file: %d\n", analysis_data.total_words_original);

    // Sentence counting using punctuation markers
    analysis_data.sentences = 0;
    int in_sentence = 0;
    for (int i = 0; analysis_data.text[i] && i < MAX_TEXT_LENGTH; i++) {
        if (analysis_data.text[i] == '.' || analysis_data.text[i] == '!' || analysis_data.text[i] == '?') {
            if (in_sentence) { analysis_data.sentences++; in_sentence = 0; }
            while (analysis_data.text[i + 1] == '.' || analysis_data.text[i + 1] == '!' || analysis_data.text[i + 1] == '?') i++;
        }
        else if (ISALPHA(analysis_data.text[i])) {
            in_sentence = 1;
        }
    }
    if (in_sentence) analysis_data.sentences++;
    if (analysis_data.sentences == 0) analysis_data.sentences = 1;

    analysis_data.text_filtered = true;

    printf("Text processing completed successfully!\n");
    printf("Original words: %d, Filtered words: %d, Sentences: %d\n",
        analysis_data.total_words_original, analysis_data.total_words_filtered, analysis_data.sentences);

    // Reset manual-save state for this session
    user_manually_saved_current_session = false;
    current_manual_filtered_filename[0] = '\0';
}

// ====== Analysis display functions for Stage 2 ======
void word_analysis() {
    if (analysis_data.text == NULL) {
        printf("No file filtered. Use option 1 first.\n");
        return;
    }

    printf("\n=== WORD STATISTICS WITH ADVANCED ANALYSIS ===\n");
    printf("File Analysed: %s\n", current_filename);
    printf("Total words                   : %d\n", analysis_data.total_words_filtered);
    printf("Unique words                  : %d\n", analysis_data.word_count);
    printf("Total sentences detected      : %d\n", analysis_data.sentences);

    if (analysis_data.sentences > 0) {
        printf("Average sentence length       : %.1f words\n",
            (float)analysis_data.total_words_filtered / analysis_data.sentences);
    }
    else {
        printf("Average sentence length       : 0.0 words\n");
    }

    printf("Total character count         : %d\n", analysis_data.total_chars);

    if (analysis_data.total_words_filtered > 0) {
        printf("Average word length           : %.1f characters\n",
            (float)analysis_data.total_chars / analysis_data.total_words_filtered);
    }
    else {
        printf("Average word length           : 0.0 characters\n");
    }

    float lexical_diversity = 0.0;
    if (analysis_data.total_words_filtered > 0) {
        lexical_diversity = (float)analysis_data.word_count / analysis_data.total_words_filtered;
    }
    printf("Lexical Diversity             : %.3f", lexical_diversity);
    if (lexical_diversity > 0.8) printf(" (High - Rich vocabulary)");
    else if (lexical_diversity > 0.6) printf(" (Medium - Good variety)");
    else if (lexical_diversity > 0.0) printf(" (Low - Repetitive vocabulary)");
    else printf(" (No vocabulary data)");
    printf("\n");

    printf("Stopwords filtered out        : %d\n", analysis_data.stopwords_removed);

    if (analysis_data.variant_processing_enabled) {
        printf("Text Normalisation            : ENABLED (expands abbreviations and Leet Speak)\n");
    }
    else {
        printf("Text Normalisation            : DISABLED (uses original text forms)\n");
    }

    // Show top 10 frequent words
    printf("\n--- TOP 10 FREQUENT WORDS ---\n");
    if (analysis_data.word_count > 0) {
        sort_by_frequency(analysis_data.words, analysis_data.word_count);
        int n = (analysis_data.word_count < 10) ? analysis_data.word_count : 10;
        for (int i = 0; i < n; i++) {
            printf("%2d. %-15s (used %d times)\n",
                i + 1, analysis_data.words[i].word, analysis_data.words[i].count);
        }
    }
    else {
        printf("No words available.\n");
    }
}

// Automatically save filtered word list silently
void save_filtered_word_list_auto(const char* filename) {
    if (analysis_data.filtered_word_count == 0 || !analysis_data.text_filtered) {
        return;
    }

    FILE* file = fopen(filename, "w");
    if (file) {
        for (int i = 0; i < analysis_data.filtered_word_count; i++) {
            if (analysis_data.filtered_word_list[i] != NULL) {
                fprintf(file, "%s\n", analysis_data.filtered_word_list[i]);
            }
        }
        fclose(file);
        // Silent save: no console message
    }
}

// Let user save filtered word list to named text file
void save_filtered_word_list() {
    if (analysis_data.text == NULL) {
        printf("No file filtered. Use option 1 first.\n");
        return;
    }

    char filename[256];
    printf("Enter filename (or press Enter for 'filtered_words.txt'): ");
    if (!read_line(filename, sizeof(filename))) {
        printf("Failed to read filename.\n");
        return;
    }

    if (strlen(filename) == 0) {
        strcpy(filename, "filtered_words.txt");
    }

    // Ensure .txt extension
    if (strlen(filename) < 4 || strcmp(filename + strlen(filename) - 4, ".txt") != 0) {
        strcat(filename, ".txt");
    }

    FILE* file = fopen(filename, "w");
    if (file) {
        // Record which source file these filtered words came from
        fprintf(file, "# SourceFile: %s\n", current_filename[0] ? current_filename : "(unknown)");

        // Record text normalisation state at save time
        fprintf(file, "# TextNormalisation: %s\n", analysis_data.variant_processing_enabled ? "enabled" : "disabled");

        for (int i = 0; i < analysis_data.filtered_word_count; i++) {
            fprintf(file, "%s\n", analysis_data.filtered_word_list[i]);
        }
        fclose(file);

        strncpy(current_manual_filtered_filename, filename, sizeof(current_manual_filtered_filename) - 1);
        current_manual_filtered_filename[sizeof(current_manual_filtered_filename) - 1] = '\0';

        // Mark that this session has been manually saved
        user_manually_saved_current_session = true;
        printf("Filtered words saved to: %s (%d words)\n", filename, analysis_data.filtered_word_count);
    }
    else {
        printf("Could not save to: %s\n", filename);
    }
}

// Simple bubble sort by frequency (descending)
void sort_by_frequency(struct WordInfo words[], int count) {
    for (int i = 0; i < count - 1; i++) {
        for (int j = 0; j < count - i - 1; j++) {
            if (words[j].count < words[j + 1].count) {
                struct WordInfo temp = words[j];
                words[j] = words[j + 1];
                words[j + 1] = temp;
            }
        }
    }
}

// Free all heap-allocated analysis buffers and reset counters
void cleanup_analysis_data() {
    if (analysis_data.text != NULL) {
        free(analysis_data.text);
        analysis_data.text = NULL;
    }
    if (analysis_data.words != NULL) {
        free(analysis_data.words);
        analysis_data.words = NULL;
    }
    if (analysis_data.filtered_word_list != NULL) {
        for (int i = 0; i < MAX_WORDS; i++) {
            if (analysis_data.filtered_word_list[i] != NULL) {
                free(analysis_data.filtered_word_list[i]);
            }
        }
        free(analysis_data.filtered_word_list);
        analysis_data.filtered_word_list = NULL;
    }
    if (analysis_data.original_word_list != NULL) {
        for (int i = 0; i < MAX_WORDS; i++) {
            if (analysis_data.original_word_list[i] != NULL) {
                free(analysis_data.original_word_list[i]);
            }
        }
        free(analysis_data.original_word_list);
        analysis_data.original_word_list = NULL;
    }

    // Reset all Stage 2 counters
    analysis_data.word_count = 0;
    analysis_data.total_words_filtered = 0;
    analysis_data.total_chars = 0;
    analysis_data.sentences = 0;
    analysis_data.stopwords_removed = 0;
    analysis_data.total_words_original = 0;
    analysis_data.filtered_word_count = 0;
    analysis_data.original_word_count = 0;
    analysis_data.text_filtered = false;

    // Reset Stage 3 toxic-related counters
    analysis_data.total_toxic_occurrences = 0;
    analysis_data.toxicity_density = 0.0;
    analysis_data.bigram_toxic_occurrences = 0;
    analysis_data.trigram_toxic_occurrences = 0;
    memset(analysis_data.severity_count, 0, sizeof(analysis_data.severity_count));
}

// Display the advanced analysis submenu
void display_advanced_analysis_menu() {
    int sub = -1;
    if (!auto_select_source_file()) {
        printf("\n[X] No files available for advanced analysis. "
            "Please load files in Stage 1 first.\n");
        return;
    }
    do {
        if (!auto_select_source_file()) {
            printf("\n[X] No files available.\n");
            return;
        }

        const char* activePath = (g_use_file == 1)
            ? inputFilePath1
            : inputFilePath2;
        printf("\n=== ADVANCED TEXT ANALYSIS MENU ===\n");
        printf("Current source file : %s (%s)\n",
            (g_use_file == 1) ? "File 1" : "File 2",
            activePath);
        printf("-----------------------------------\n");
        printf("1. Change source file\n");
        printf("2. Word Analysis\n");
        printf("3. Text Normalisation (Toggle & View Examples)\n");
        printf("4. Save Filtered Word List\n");
        printf("0. Back\n");
        printf("Select: ");

        if (scanf("%d", &sub) != 1) {
            int c;
            while ((c = getchar()) != '\n' && c != EOF);
            printf("Invalid input. Please enter 0-4.\n");
            sub = -1;
            continue;
        }

        int c;
        while ((c = getchar()) != '\n' && c != EOF);

        switch (sub) {
        case 1:
            prompt_change_source_file();
            break;
        case 2: {
            // Extra safety: ensure a valid source is available
            if (!auto_select_source_file()) {
                printf("[X] No files available. Please load files in Stage 1 first.\n");
                break;
            }
            activePath = (g_use_file == 1) ? inputFilePath1 : inputFilePath2;

            printf("Processing %s: %s\n",
                (g_use_file == 1) ? "File 1" : "File 2",
                activePath);
            // This will clear old analysis_data and re-run Stage 2
            process_text_file(activePath);
            word_analysis();
        } break;
        case 3:
            if (!analysis_data.text_filtered) {
                printf("No analysis available. Please use option 2 first.\n");
            }
            else {
                // Combined functionality: toggle + show examples
                toggle_variant_processing();
            }
            break;
        case 4:
            if (!analysis_data.text_filtered) {
                printf("No analysis available. Please use option 2 first.\n");
            }
            else {
                save_filtered_word_list();
            }
            break;
        case 0:
            printf("Returning to main menu...\n");
            break;
        default:
            printf("Invalid choice. Please enter 0-4.\n");
        }
    } while (sub != 0);
}



// ===== 3. Stage 3 - Toxic Dictionary Management and Toxicity Analysis =====

// Load toxic words from toxicwords.txt into the in-memory dictionary (Stage 4).
static int load_toxicwords(void) {
    if (g_toxic_loaded) return g_toxic_count;
    FILE* f = fopen("toxicwords.txt", "r");
    if (!f) { printf("[!] Cannot open toxicwords.txt (toxic TopN will be empty)\n"); return 0; }
    char line[MAX_WORD_LENGTH];
    while (g_toxic_count < 500 && fgets(line, sizeof(line), f)) {
        line[strcspn(line, "\r\n")] = '\0';
        for (int i = 0; line[i]; ++i) line[i] = (char)tolower((unsigned char)line[i]);
        if (line[0]) strcpy(g_toxic[g_toxic_count++], line);
    }
    fclose(f);
    g_toxic_loaded = 1;
    printf("[i] Loaded %d toxic terms\n", g_toxic_count);
    return g_toxic_count;
}

// Sync toxic words between Stage 3 and Stage 4 systems
void sync_toxic_systems(void) {
    // Reset Stage 4 system
    g_toxic_count = 0;
    g_toxic_loaded = 0;

    // Copy from Stage 3 to Stage 4
    for (int i = 0; i < analysis_data.toxic_words_count && g_toxic_count < 500; i++) {
        strcpy(g_toxic[g_toxic_count], analysis_data.toxic_words_list[i].word);
        g_toxic_count++;
    }
    g_toxic_loaded = 1;
    printf("[i] Synced %d toxic terms between systems\n", g_toxic_count);
}

// Load toxic words and phrases from a dictionary file.
void load_toxic_data(const char* filename) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        printf("Warning: Cannot open toxic data file: %s\n", filename);
        return;
    }

    analysis_data.toxic_words_count = 0;
    analysis_data.toxic_phrases_count = 0;
    char line[256];

    printf("Loading toxic data from %s...\n", filename);

    while (fgets(line, sizeof(line), file)) {
        line[strcspn(line, "\n")] = '\0';
        line[strcspn(line, "\r")] = '\0';

        if (strlen(line) == 0 || line[0] == '#') continue;

        char* token = strtok(line, ",");
        if (!token) continue;

        char word[100];
        strcpy(word, token);

        trim_inplace(word);
        tolower_inplace(word);

        token = strtok(NULL, ",");
        if (!token) {
            printf("Warning: Invalid line (no severity): %s\n", line);
            continue;
        }
        trim_inplace(token);
        int severity = atoi(token);
        if (severity < 1 || severity > 5) severity = 3;

        if (strchr(word, ' ') != NULL) {
            if (analysis_data.toxic_phrases_count < MAX_PHRASES) {
                int idx = analysis_data.toxic_phrases_count;

                int len = 1;
                for (char* p = word; *p; ++p) {
                    if (*p == ' ') len++;
                }

                strncpy(analysis_data.toxic_phrases_list[idx].phrase,
                    word, MAX_WORD_LENGTH * 3 - 1);
                analysis_data.toxic_phrases_list[idx].phrase[MAX_WORD_LENGTH * 3 - 1] = '\0';

                analysis_data.toxic_phrases_list[idx].severity = severity;
                analysis_data.toxic_phrases_list[idx].frequency = 0;

                if (len == 2 || len == 3) {
                    analysis_data.toxic_phrases_list[idx].ngram_len = len;
                }
                else {
                    analysis_data.toxic_phrases_list[idx].ngram_len = 0;
                }

                analysis_data.toxic_phrases_count++;
            }
        }
        else {
            if (analysis_data.toxic_words_count < MAX_TOXIC_WORDS) {
                int idx = analysis_data.toxic_words_count;
                strncpy(analysis_data.toxic_words_list[idx].word,
                    word, MAX_WORD_LENGTH - 1);
                analysis_data.toxic_words_list[idx].word[MAX_WORD_LENGTH - 1] = '\0';
                analysis_data.toxic_words_list[idx].severity = severity;
                analysis_data.toxic_words_list[idx].frequency = 0;
                analysis_data.toxic_words_count++;
            }
        }
    }

    fclose(file);
    printf("Loaded %d toxic words and %d toxic phrases from %s\n",
        analysis_data.toxic_words_count, analysis_data.toxic_phrases_count, filename);
    sync_toxic_systems();
}

// Check if a word is toxic (works for both Stage 3 and Stage 4)
int is_toxic_word(const char* word) {
    if (!word || !*word) return 0;

    char tmp[MAX_WORD_LENGTH];
    strncpy(tmp, word, MAX_WORD_LENGTH - 1);
    tmp[MAX_WORD_LENGTH - 1] = '\0';

    size_t len = strlen(tmp);
    while (len > 0 && isspace((unsigned char)tmp[len - 1])) {
        tmp[--len] = '\0';
    }

    for (size_t i = 0; i < len; ++i) {
        tmp[i] = (char)tolower((unsigned char)tmp[i]);
    }

    // Check Stage 3 system first
    for (int i = 0; i < analysis_data.toxic_words_count; i++) {
        if (string_case_insensitive_compare(tmp, analysis_data.toxic_words_list[i].word) == 0) {
            return 1;
        }
    }

    // 3) Make sure Dictionary of Stage 4 already be loaded
    if (!g_toxic_loaded) {
        load_toxicwords();
    }

    // Also check Stage 4 system as backup
    for (int i = 0; i < g_toxic_count; i++) {
        if (string_case_insensitive_compare(tmp, g_toxic[i]) == 0) {
            return 1;
        }
    }

    return 0;
}

// Retrieve the defined severity level (1–5) of a toxic word.
// Returns 0 if the word is not classified as toxic.
int get_toxic_severity(const char* word) {
    for (int i = 0; i < analysis_data.toxic_words_count; i++) {
        if (string_case_insensitive_compare(word, analysis_data.toxic_words_list[i].word) == 0) {
            return analysis_data.toxic_words_list[i].severity;
        }
    }
    return 0;
}

// Return the index of a toxic word in the internal dictionary.
// Returns -1 if not found.
int find_toxic_index(const char* word) {
    for (int i = 0; i < analysis_data.toxic_words_count; i++) {
        if (string_case_insensitive_compare(word, analysis_data.toxic_words_list[i].word) == 0)
            return i;
    }
    return -1;
}

// Update frequency and severity statistics for a toxic word occurrence.
void detect_toxic_content(const char* word) {
    if (is_toxic_word(word)) {
        analysis_data.total_toxic_occurrences++;
        int severity = get_toxic_severity(word);

        for (int i = 0; i < analysis_data.toxic_words_count; i++) {
            if (string_case_insensitive_compare(word, analysis_data.toxic_words_list[i].word) == 0) {
                analysis_data.toxic_words_list[i].frequency++;
                break;
            }
        }

        if (severity >= 1 && severity <= 5) {
            analysis_data.severity_count[severity]++;
        }
    }
}

// Detect toxic phrases (2-gram or 3-gram) formed by consecutive words.
// Matches phrases defined in the dictionary and updates frequency counts.
void detect_toxic_phrases() {
    if (analysis_data.original_word_count < 2) return;

    for (int i = 0; i <= analysis_data.original_word_count - 2; i++) {

        // ==== 2-gram ====
        char phrase2[MAX_WORD_LENGTH * 3] = "";
        strncat(phrase2, analysis_data.original_word_list[i], MAX_WORD_LENGTH);
        strcat(phrase2, " ");
        strncat(phrase2, analysis_data.original_word_list[i + 1], MAX_WORD_LENGTH);

        for (int j = 0; j < analysis_data.toxic_phrases_count; j++) {
            if (analysis_data.toxic_phrases_list[j].ngram_len != 2) continue;

            if (string_case_insensitive_compare(phrase2,
                analysis_data.toxic_phrases_list[j].phrase) == 0) {

                analysis_data.toxic_phrases_list[j].frequency++;
                analysis_data.bigram_toxic_occurrences++;

                break;
            }
        }

        // ==== 3-gram ====
        if (i <= analysis_data.original_word_count - 3) {
            char phrase3[MAX_WORD_LENGTH * 3] = "";
            strncat(phrase3, analysis_data.original_word_list[i], MAX_WORD_LENGTH);
            strcat(phrase3, " ");
            strncat(phrase3, analysis_data.original_word_list[i + 1], MAX_WORD_LENGTH);
            strcat(phrase3, " ");
            strncat(phrase3, analysis_data.original_word_list[i + 2], MAX_WORD_LENGTH);

            for (int j = 0; j < analysis_data.toxic_phrases_count; j++) {
                if (analysis_data.toxic_phrases_list[j].ngram_len != 3) continue;

                if (string_case_insensitive_compare(phrase3,
                    analysis_data.toxic_phrases_list[j].phrase) == 0) {

                    analysis_data.toxic_phrases_list[j].frequency++;
                    analysis_data.trigram_toxic_occurrences++;

                    break;
                }
            }
        }
    }
}

// Analyse a phrase to determine how many toxic words it contains,
// store them in the provided buffer, and return the maximum severity.
int phrase_contains_toxic_words(const char* phrase, char found_words[][MAX_WORD_LENGTH], int max_found, int* max_severity) {
    if (!phrase || !*phrase) return 0;

    char phrase_copy[MAX_WORD_LENGTH * 3];
    strcpy(phrase_copy, phrase);

    int found_count = 0;
    int local_max_sev = 0;

    char* word = strtok(phrase_copy, " ");
    while (word != NULL && found_count < max_found) {
        if (is_toxic_word(word)) {
            strcpy(found_words[found_count], word);
            int sev = get_toxic_severity(word);
            if (sev > local_max_sev) local_max_sev = sev;
            found_count++;
        }
        word = strtok(NULL, " ");
    }

    if (max_severity) {
        *max_severity = local_max_sev;
    }
    return found_count;
}

// Reset all toxicity-related counters before performing a new analysis run.
void reset_toxic_counts() {
    analysis_data.total_toxic_occurrences = 0;
    analysis_data.toxicity_density = 0.0;
    memset(analysis_data.severity_count, 0, sizeof(analysis_data.severity_count));
    analysis_data.bigram_toxic_occurrences = 0;
    analysis_data.trigram_toxic_occurrences = 0;

    for (int i = 0; i < analysis_data.toxic_words_count; i++) {
        analysis_data.toxic_words_list[i].frequency = 0;
    }
    for (int i = 0; i < analysis_data.toxic_phrases_count; i++) {
        analysis_data.toxic_phrases_list[i].frequency = 0;
    }
}

// Compute toxicity density as a percentage of toxic words among all filtered words.
void calculate_toxicity_density() {
    if (analysis_data.total_words_filtered > 0) {
        analysis_data.toxicity_density = (float)analysis_data.total_toxic_occurrences /
            analysis_data.total_words_filtered * 100;
    }
    else {
        analysis_data.toxicity_density = 0.0;
    }
}

// Execute the full toxic detection pipeline.
// Handles normalisation, word-list selection, toxicity scanning,
// phrase detection, and final density computation.
void run_toxic_analysis(void) {
    if (!analysis_data.text_filtered) {
        printf("No text filtered. Please load and process a file first.\n");
        return;
    }

    char filename_to_use[256] = "";
    bool need_reprocess = false;
    bool manual_source_match = false;
    bool saved_without_normalisation = false;

    const char* manual_name =
        (strlen(current_manual_filtered_filename) > 0)
        ? current_manual_filtered_filename
        : "filtered_words.txt";

    if (user_manually_saved_current_session) {
        FILE* f = fopen(manual_name, "r");
        if (f) {
            char line[512];
            char source_path[256] = "";

            while (fgets(line, sizeof(line), f)) {
                line[strcspn(line, "\r\n")] = 0;
                if (line[0] != '#') break;

                if (strstr(line, "TextNormalisation: disabled"))
                    saved_without_normalisation = true;
                else if (strstr(line, "TextNormalisation: enabled"))
                    saved_without_normalisation = false;
                else if (!strncmp(line, "# SourceFile:", 13)) {
                    const char* p = line + 13;
                    while (*p == ' ' || *p == '\t') p++;
                    strncpy(source_path, p, sizeof(source_path) - 1);
                    source_path[sizeof(source_path) - 1] = '\0';
                }
            }

            if (source_path[0] && current_filename[0] &&
                strcmp(source_path, current_filename) == 0) {
                manual_source_match = true;
            }
            fclose(f);
        }
    }

    bool using_manual_file = user_manually_saved_current_session && manual_source_match;

    if (using_manual_file) {
        if (saved_without_normalisation) {
            printf("\nWARNING: Your filtered word list was saved WITHOUT Text Normalisation.\n");
            printf("This means abbreviations and Leet Speak won't be expanded.\n");
            printf("Do you want to:\n");
            printf("1. Use the non-normalised file (faster, less accurate)\n");
            printf("2. Re-process with Text Normalisation (recommended)\n");
            printf("Enter your option (1-2): ");

            char option;
            scanf(" %c", &option);

            if (option == '1') {
                printf("\nUsing non-normalised file for analysis: %s\n", manual_name);
                strcpy(filename_to_use, manual_name);
            }
            else {
                printf("\nRe-processing with Text Normalisation...\n");
                if (!analysis_data.variant_processing_enabled) {
                    analysis_data.variant_processing_enabled = true;
                    need_reprocess = true;
                }
                strcpy(filename_to_use, "filtered_words_normalised.txt");
            }
        }
        else {
            printf("Using your manually saved file: %s\n", manual_name);
            strcpy(filename_to_use, manual_name);
        }
    }
    else {
        printf("No matching manually saved filtered word list for this file.\n");
        printf("Auto-saving normalised word list for toxic analysis...\n");

        if (!analysis_data.variant_processing_enabled) {
            printf("Enabling Text Normalisation for better accuracy...\n");
            analysis_data.variant_processing_enabled = true;
            need_reprocess = true;
        }
        strcpy(filename_to_use, "filtered_words_auto_saved.txt");
    }

    if (need_reprocess) {
        reprocess_with_variants();
    }

    if (!using_manual_file ||
        strcmp(filename_to_use, "filtered_words_normalised.txt") == 0) {
        save_filtered_word_list_auto(filename_to_use);
        printf("Auto-saved word list: %s\n", filename_to_use);
    }

    reset_toxic_counts();
    printf("Starting toxic analysis using: %s\n", filename_to_use);

    FILE* file = fopen(filename_to_use, "r");
    if (!file) {
        printf("Error: Cannot open filtered word list file: %s\n", filename_to_use);
        return;
    }

    char word[MAX_WORD_LENGTH];
    int word_count = 0;

    while (fgets(word, sizeof(word), file) && word_count < MAX_WORDS) {
        word[strcspn(word, "\r\n")] = 0;
        if (strlen(word) > 0) {
            detect_toxic_content(word);
            word_count++;
        }
    }
    fclose(file);

    printf("Analysed %d words from file\n", word_count);

    detect_toxic_phrases();
    calculate_toxicity_density();
    printf("Toxic analysis completed.\n");
}

// Entry point for Stage 3 toxic content inspection.
// Loads dictionary if missing and prints detailed toxicity report.
void toxic_analysis() {
    if (analysis_data.toxic_words_count == 0 && analysis_data.toxic_phrases_count == 0) {
        printf("Loading toxic dictionary...\n");
        load_toxic_data("toxicwords.txt");
    }

    printf("\n=== TOXIC CONTENT ANALYSIS ===\n");
    printf("Detecting for toxic content...\n");
    run_toxic_analysis();

    if (analysis_data.total_toxic_occurrences == 0) {
        printf("Your file contains no toxic content.\n");
        return;
    }

    printf("\n--- TOXIC CONTENT SUMMARY ---\n");
    printf("Total toxic words detected: %d\n", analysis_data.total_toxic_occurrences);
    printf("Toxicity score: %.2f%% (%d toxic words out of %d total words detected)\n",
        analysis_data.toxicity_density,
        analysis_data.total_toxic_occurrences,
        analysis_data.total_words_filtered);

    // All scores are computed at word level (single-word detections).
    printf(" - Word-level (single words) : %d detections\n",
        analysis_data.total_toxic_occurrences);

    // Phrase detections (bigrams/trigrams) are reported separately and explicitly not included in the main toxicity score.
    printf(" - Toxic bigram matches      : %d detections (not counted in total)\n",
        analysis_data.bigram_toxic_occurrences);
    printf(" - Toxic trigram matches     : %d detections (not counted in total)\n",
        analysis_data.trigram_toxic_occurrences);

    // Show severity distribution using a simple text-based bar chart.
    printf("\n--- SEVERITY DISTRIBUTION ---\n");
    int max_count = 0;
    for (int i = 1; i <= 5; i++) {
        if (analysis_data.severity_count[i] > max_count) {
            max_count = analysis_data.severity_count[i];
        }
    }

    for (int i = 1; i <= 5; i++) {
        if (analysis_data.severity_count[i] > 0) {
            float percentage = (float)analysis_data.severity_count[i] / analysis_data.total_toxic_occurrences * 100;
            int bar_length = max_count > 0 ? (analysis_data.severity_count[i] * 20 / max_count) : 0;

            printf("Level %d: ", i);
            for (int j = 0; j < bar_length; j++) printf("#");
            printf(" %d words (%.1f%%)\n", analysis_data.severity_count[i], percentage);
        }
    }

    // List toxic words detected in this analysis, sorted by frequency.
    printf("\n--- TOXIC WORDS DETECTED ---\n");

    struct ToxicWord sorted_words[MAX_TOXIC_WORDS];
    int valid_count = 0;

    for (int i = 0; i < analysis_data.toxic_words_count; i++) {
        if (analysis_data.toxic_words_list[i].frequency > 0) {
            sorted_words[valid_count] = analysis_data.toxic_words_list[i];
            valid_count++;
        }
    }

    // Sort the temporary array in descending order of frequency (simple bubble sort).
    for (int i = 0; i < valid_count - 1; i++) {
        for (int j = 0; j < valid_count - i - 1; j++) {
            if (sorted_words[j].frequency < sorted_words[j + 1].frequency) {
                struct ToxicWord temp = sorted_words[j];
                sorted_words[j] = sorted_words[j + 1];
                sorted_words[j + 1] = temp;
            }
        }
    }

    if (valid_count > 0) {
        printf("+-----------------+-----------+--------------+\n");
        printf("| Word            | Frequency | Level (1-5)  |\n");
        printf("+-----------------+-----------+--------------+\n");

        for (int i = 0; i < valid_count; i++) {
            // Compute how many digits the frequency has, so we can centre it in the column.
            int freq = sorted_words[i].frequency;
            int freq_digits = 0;
            if (freq == 0) freq_digits = 1;
            else {
                int temp = freq;
                while (temp > 0) {
                    freq_digits++;
                    temp /= 10;
                }
            }

            // Compute padding before and after the number to centre it (column width 11).
            int total_spaces = 11 - freq_digits;
            int spaces_before = total_spaces / 2;
            int spaces_after = total_spaces - spaces_before;

            printf("| %-15s |", sorted_words[i].word);

            for (int s = 0; s < spaces_before; s++) printf(" ");
            printf("%d", freq);
            for (int s = 0; s < spaces_after; s++) printf(" ");

            printf("|      %d       |\n", sorted_words[i].severity);
        }
        printf("+-----------------+-----------+--------------+\n");
    }
    else {
        printf("No toxic words found.\n");
    }

    // Report toxic phrase patterns that were matched in the text.
    printf("\n--- TOXIC PHRASES (patterns only) ---\n");
    int phrase_found = 0;
    int total_phrase_occurrences = 0;

    //  First compute the total number of toxic phrase occurrences (for summary).
    for (int i = 0; i < analysis_data.toxic_phrases_count; i++) {
        if (analysis_data.toxic_phrases_list[i].frequency > 0) {
            total_phrase_occurrences += analysis_data.toxic_phrases_list[i].frequency;
        }
    }

    if (total_phrase_occurrences > 0) {
        for (int i = 0; i < analysis_data.toxic_phrases_count; i++) {
            if (analysis_data.toxic_phrases_list[i].frequency > 0) {

                const char* phrase = analysis_data.toxic_phrases_list[i].phrase;
                int freq = analysis_data.toxic_phrases_list[i].frequency;
                int sev = analysis_data.toxic_phrases_list[i].severity;

                printf("Detected phrase: %s\n", phrase);
                printf("  Frequency: %d time(s)\n", freq);

                // Use helper to inspect how many toxic words appear inside the phrase.
                char found_words[10][MAX_WORD_LENGTH];
                int max_sev_in_phrase = 0;
                int toxic_word_count = phrase_contains_toxic_words(
                    phrase,
                    found_words,
                    10,
                    &max_sev_in_phrase
                );

                if (toxic_word_count > 0) {
                    printf("  Contains toxic word(s): ");
                    for (int k = 0; k < toxic_word_count; k++) {
                        if (k > 0) printf(", ");
                        printf("%s", found_words[k]);
                    }
                    printf("\n");

                    if (toxic_word_count >= 2) {
                        // Fixed combinations containing multiple toxic words (multi-toxic phrase).
                        printf("  Note: Multi-toxic phrase (contains %d toxic words, max severity %d)\n",
                            toxic_word_count,
                            max_sev_in_phrase > 0 ? max_sev_in_phrase : sev);
                    }
                }
                else {
                    printf("  Context-based toxicity (combination of non-toxic words)\n");
                }

                printf("  Overall severity: Level %d\n\n", sev);
                phrase_found = 1;
            }
        }
    }
    else {
        printf("No toxic phrases detected.\n");
    }
}

// Display all toxic words and phrases currently stored in the dictionary.
void view_all_toxic_words() {
    printf("\n=== TOXIC DICTIONARY OVERVIEW ===\n");
    printf("Total words: %d, Total phrases: %d\n\n",
        analysis_data.toxic_words_count, analysis_data.toxic_phrases_count);

    // Show words grouped by severity level.
    printf("TOXIC WORDS BY SEVERITY LEVEL:\n");
    printf("-------------------------------\n");

    for (int severity = 1; severity <= 5; severity++) {
        printf("\nLevel %d:\n", severity);
        int count = 0;
        for (int i = 0; i < analysis_data.toxic_words_count; i++) {
            if (analysis_data.toxic_words_list[i].severity == severity) {
                printf("%-15s", analysis_data.toxic_words_list[i].word);
                count++;
                if (count % 5 == 0) printf("\n"); // Print 5 words per line.
            }
        }
        if (count % 5 != 0) printf("\n"); // Ensure ending newline for partial lines.
        printf("Total: %d words\n", count);
    }

    // Show all toxic phrases and their related toxic words.
    printf("\nTOXIC PHRASES:\n");
    printf("--------------\n");
    int phrases_displayed = 0;
    for (int i = 0; i < analysis_data.toxic_phrases_count; i++) {
        // Check if the phrase contains any known toxic words.
        char found_toxic_words[10][MAX_WORD_LENGTH];
        int toxic_word_count = phrase_contains_toxic_words(
            analysis_data.toxic_phrases_list[i].phrase,
            found_toxic_words,
            10,
            NULL   // No need to track max severity here.
        );

        printf("%2d. %s (Level %d)",
            i + 1,
            analysis_data.toxic_phrases_list[i].phrase,
            analysis_data.toxic_phrases_list[i].severity);

        if (toxic_word_count > 0) {
            printf(" - Contains: ");
            for (int j = 0; j < toxic_word_count; j++) {
                if (j > 0) printf(", ");
                printf("%s", found_toxic_words[j]);
            }
        }
        else {
            printf(" - Combination toxicity");
        }
        printf("\n");
        phrases_displayed++;
    }
    if (phrases_displayed == 0) {
        printf("No toxic phrases in dictionary.\n");
    }
}

// Menu handler for dictionary operations (add/remove/view toxic entries).
void dictionary_management() {
    char option;
    do {
        printf("\n=== TOXIC DICTIONARY MANAGEMENT ===\n");
        printf("Current dictionary: %d words and %d phrases\n",
            analysis_data.toxic_words_count, analysis_data.toxic_phrases_count);
        printf("File: toxicwords.txt\n");

        printf("\n1. Add new toxic word or phrase\n");
        printf("2. Remove toxic word or phrase\n");
        printf("3. View dictionary\n");
        printf("0. Back to menu\n");
        option = get_menu_option("1230", "Enter your option (0-3): ");

        switch (option) {
        case '1':
            add_custom_toxic_word();
            break;
        case '2':
            remove_toxic_word();
            break;
        case '3':
            view_all_toxic_words();
            break;
        case '0':
            printf("Returning to menu...\n");
            break;
        default:
            printf("Invalid option. Please enter 0-3.\n");
        }
    } while (option != '0');
}

// Add a custom toxic word or phrase to the in-memory dictionary and save to file.
void add_custom_toxic_word() {
    if (analysis_data.toxic_words_count >= MAX_TOXIC_WORDS - 1) {
        printf("Toxic words list is full!\n");
        return;
    }

    char new_input[MAX_WORD_LENGTH * 3];
    printf("Enter new toxic word or phrase: ");
    if (!read_line(new_input, sizeof(new_input))) {
        printf("Failed to read input.\n");
        return;
    }

    // Normalise input to lowercase for matching and storage.
    char lower_input[MAX_WORD_LENGTH * 3];
    strcpy(lower_input, new_input);
    for (int i = 0; lower_input[i]; i++) {
        lower_input[i] = (char)tolower((unsigned char)lower_input[i]);
    }

    int i = 0;
    while (lower_input[i] && isspace((unsigned char)lower_input[i])) i++;
    if (lower_input[i] == '\0') return;

    // Decide whether this is a single word or a phrase (contains spaces).
    if (strchr(lower_input, ' ') == NULL) {
        // ===== Single-word branch =====
        // Check if word already exists in the dictionary.
        for (int i = 0; i < analysis_data.toxic_words_count; i++) {
            if (string_case_insensitive_compare(lower_input,
                analysis_data.toxic_words_list[i].word) == 0) {
                printf("Word '%s' already exists in the dictionary.\n", new_input);
                return;
            }
        }

        int severity;
        printf("Enter severity (Level 1-5): ");
        scanf("%d", &severity);
        getchar();
        if (severity < 1 || severity > 5) {
            printf("Invalid severity. Using Level 3 as default.\n");
            severity = 3;
        }

        // Insert the new word into the list, keeping alphabetical order.
        int insert_pos = analysis_data.toxic_words_count;
        for (int i = 0; i < analysis_data.toxic_words_count; i++) {
            if (string_case_insensitive_compare(lower_input,
                analysis_data.toxic_words_list[i].word) < 0) {
                insert_pos = i;
                break;
            }
        }
        for (int i = analysis_data.toxic_words_count; i > insert_pos; i--) {
            analysis_data.toxic_words_list[i] = analysis_data.toxic_words_list[i - 1];
        }

        strcpy(analysis_data.toxic_words_list[insert_pos].word, lower_input);
        analysis_data.toxic_words_list[insert_pos].severity = severity;
        analysis_data.toxic_words_list[insert_pos].frequency = 0;
        analysis_data.toxic_words_count++;

        save_toxic_dictionary("toxicwords.txt");
        printf("Added word '%s' with Level %d and saved to dictionary\n",
            new_input, severity);
    }
    else {
        // ===== phrase branch =====
        add_custom_toxic_phrase(lower_input);
    }
}

// Add a new toxic phrase and optionally mark words inside it as toxic words.
void add_custom_toxic_phrase(const char* phrase) {
    if (analysis_data.toxic_phrases_count >= MAX_PHRASES) {
        printf("Toxic phrase list is full!\n");
        return;
    }

    // Split the phrase into individual words (up to 10).
    char phrase_copy[MAX_WORD_LENGTH * 3];
    strncpy(phrase_copy, phrase, sizeof(phrase_copy) - 1);
    phrase_copy[sizeof(phrase_copy) - 1] = '\0';

    char words[10][MAX_WORD_LENGTH];
    int  word_count = 0;

    char* tok = strtok(phrase_copy, " ");
    while (tok && word_count < 10) {
        strncpy(words[word_count], tok, MAX_WORD_LENGTH - 1);
        words[word_count][MAX_WORD_LENGTH - 1] = '\0';
        word_count++;
        tok = strtok(NULL, " ");
    }

    if (word_count < 2) {
        printf("Phrase should contain at least two words.\n");
        return;
    }

    // Check which words are already in the toxic word dictionary.
    int is_toxic[10] = { 0 };
    int word_sev[10] = { 0 };
    for (int i = 0; i < word_count; i++) {
        int sev = get_toxic_severity(words[i]);
        if (sev > 0) {
            is_toxic[i] = 1;
            word_sev[i] = sev;
        }
    }

    // Show each word in the phrase and current toxicity information (if any).
    printf("\nWords in phrase:\n");
    for (int i = 0; i < word_count; i++) {
        if (is_toxic[i]) {
            printf("  %d) %s  (already toxic, Level %d)\n",
                i + 1, words[i], word_sev[i]);
        }
        else {
            printf("  %d) %s\n", i + 1, words[i]);
        }
    }

    // Ask once which positions should be marked as toxic words (avoid many y/n prompts).
    int selected[10] = { 0 };
    char line[128];

    printf("\nWhich positions are TOXIC words? \n");
    printf("Enter numbers separated by space (e.g. 2 3), or 0 if you don't want to add new toxic words:\n> ");

    if (read_line(line, sizeof(line))) {
        char* p = strtok(line, " ");
        while (p) {
            int idx = atoi(p);
            if (idx == 0) {
                // 0 means "no new toxic words to add"
                break;
            }
            if (idx >= 1 && idx <= word_count) {
                selected[idx - 1] = 1;
            }
            p = strtok(NULL, " ");
        }
    }

    // For each selected word that is not yet toxic, ask for its severity and add it.
    for (int i = 0; i < word_count; i++) {
        if (selected[i] && !is_toxic[i]) {
            int sev;
            printf("Set severity for new toxic word '%s' (1-5): ", words[i]);
            scanf("%d", &sev);
            getchar();
            if (sev < 1 || sev > 5) {
                printf("  Invalid severity. Using Level 3.\n");
                sev = 3;
            }

            // Insert into toxic word dictionary if there is space.
            if (analysis_data.toxic_words_count < MAX_TOXIC_WORDS) {
                int idx = analysis_data.toxic_words_count;
                strcpy(analysis_data.toxic_words_list[idx].word, words[i]);
                analysis_data.toxic_words_list[idx].severity = sev;
                analysis_data.toxic_words_list[idx].frequency = 0;
                analysis_data.toxic_words_count++;
            }

            is_toxic[i] = 1;
            word_sev[i] = sev;
        }
    }

    // Finally, compute how many toxic words are in the phrase and their maximum severity.
    int final_toxic_count = 0;
    int final_max_sev = 0;
    for (int i = 0; i < word_count; i++) {
        int sev = get_toxic_severity(words[i]);
        if (sev > 0) {
            final_toxic_count++;
            if (sev > final_max_sev) final_max_sev = sev;
        }
    }

    int phrase_severity = 0;

    if (final_toxic_count == 0) {
        // Case A: purely context-based phrase (no toxic words inside).
        // Ask user for a severity for the whole phrase and store it.
        printf("\nNo toxic words inside this phrase so far.\n");
        printf("Set severity for the phrase '%s' (1-5): ", phrase);
        scanf("%d", &phrase_severity);
        getchar();
        if (phrase_severity < 1 || phrase_severity > 5) {
            printf("Invalid severity. Using Level 3.\n");
            phrase_severity = 3;
        }
    }
    else if (final_toxic_count == 1) {
        // Case B: exactly one toxic word in the phrase.
        // By design this is NOT stored as a separate toxic phrase,
        // detection will rely on that single toxic word.
        printf("\nThis phrase contains exactly ONE toxic word.\n");
        printf("It will not be stored as a toxic phrase.\n");
        printf("Toxic detection will rely on the toxic word itself only.\n");
        save_toxic_dictionary("toxicwords.txt");
        return;  // Do not add to phrase list.
    }

    else {
        // Case C: phrase contains two or more toxic words → treat as a multi-toxic phrase.
        phrase_severity = (final_max_sev > 0) ? final_max_sev : 3;
        printf("\nPhrase contains %d toxic word(s). Phrase severity set to Level %d (max of words).\n",
            final_toxic_count, phrase_severity);
    }

    // Only Case A and Case C reach this point: store phrase in dictionary.
    if (analysis_data.toxic_phrases_count < MAX_PHRASES) {
        int idx = analysis_data.toxic_phrases_count;

        strncpy(analysis_data.toxic_phrases_list[idx].phrase,
            phrase, MAX_WORD_LENGTH * 3 - 1);
        analysis_data.toxic_phrases_list[idx].phrase[MAX_WORD_LENGTH * 3 - 1] = '\0';

        analysis_data.toxic_phrases_list[idx].severity = phrase_severity;
        analysis_data.toxic_phrases_list[idx].frequency = 0;

        if (word_count == 2 || word_count == 3) {
            analysis_data.toxic_phrases_list[idx].ngram_len = word_count;
        }
        else {
            analysis_data.toxic_phrases_list[idx].ngram_len = 0;
        }

        analysis_data.toxic_phrases_count++;

        save_toxic_dictionary("toxicwords.txt");
        printf("Added phrase '%s' (severity: %d, words: %d, toxic_words: %d)\n",
            phrase, phrase_severity, word_count, final_toxic_count);
    }
}

// Remove a toxic word or phrase from the dictionary by text match.
void remove_toxic_word() {
    // Use a larger buffer so both words and phrases can be entered.
    char word_to_remove[MAX_WORD_LENGTH * 3];
    printf("Enter toxic word or phrase to remove: ");
    if (!read_line(word_to_remove, sizeof(word_to_remove))) {
        printf("Failed to read input.\n");
        return;
    }

    if (strlen(word_to_remove) == 0) {
        printf("No input provided.\n");
        return;
    }

    // Normalise to lowercase so we can do case-insensitive comparison.
    for (int i = 0; word_to_remove[i]; i++) {
        word_to_remove[i] = (char)tolower((unsigned char)word_to_remove[i]);
    }

    bool removed = false;

    // 1. Try to remove from the toxic word list first.
    for (int i = 0; i < analysis_data.toxic_words_count; i++) {
        if (string_case_insensitive_compare(word_to_remove,
            analysis_data.toxic_words_list[i].word) == 0) {

            // Shift remaining entries left to fill the gap.
            for (int j = i; j < analysis_data.toxic_words_count - 1; j++) {
                analysis_data.toxic_words_list[j] = analysis_data.toxic_words_list[j + 1];
            }
            analysis_data.toxic_words_count--;
            removed = true;

            printf("Removed '%s' from toxic word list.\n", word_to_remove);
            break;
        }
    }

    // 2. If not found in words, try to remove from the phrase list.
    if (!removed) {
        for (int i = 0; i < analysis_data.toxic_phrases_count; i++) {
            if (string_case_insensitive_compare(word_to_remove,
                analysis_data.toxic_phrases_list[i].phrase) == 0) {

                // Shift remaining entries left to fill the gap.
                for (int j = i; j < analysis_data.toxic_phrases_count - 1; j++) {
                    analysis_data.toxic_phrases_list[j] = analysis_data.toxic_phrases_list[j + 1];
                }
                analysis_data.toxic_phrases_count--;
                removed = true;

                printf("Removed '%s' from toxic phrase list.\n", word_to_remove);
                break;
            }
        }
    }

    // 3. If not found in either list, report to the user.
    if (!removed) {
        printf("'%s' not found in toxic word or phrase dictionary.\n", word_to_remove);
        return;
    }

    // If something was removed, persist changes to the backing file.
    save_toxic_dictionary("toxicwords.txt");
    printf("Dictionary file updated.\n");
}

// Save the current toxic word and phrase dictionary to a file.
void save_toxic_dictionary(const char* filename) {
    FILE* file = fopen(filename, "w");
    if (!file) {
        printf("Error: Cannot save toxic dictionary to %s\n", filename);
        return;
    }

    // Write file header.
    fprintf(file, "# Toxic Words Dictionary\n");
    fprintf(file, "# Format: word,severity\n");
    fprintf(file, "# Severity: 1-5 (1=mild, 5=severe)\n\n");

    // Write all toxic words (including custom user-added entries).
    for (int i = 0; i < analysis_data.toxic_words_count; i++) {
        fprintf(file, "%s,%d\n",
            analysis_data.toxic_words_list[i].word,
            analysis_data.toxic_words_list[i].severity);
    }

    // Write all toxic phrases.
    for (int i = 0; i < analysis_data.toxic_phrases_count; i++) {
        fprintf(file, "%s,%d\n",
            analysis_data.toxic_phrases_list[i].phrase,
            analysis_data.toxic_phrases_list[i].severity);
    }

    fclose(file);
    printf("Toxic dictionary saved to: %s (%d words, %d phrases)\n",
        filename, analysis_data.toxic_words_count, analysis_data.toxic_phrases_count);
    sync_toxic_systems();
}

// ========== STAGE 3 MENU DISPLAY FUNCTION ==========
void display_toxic_menu() {
    int sub = -1;
    do {
        if (!auto_select_source_file()) {
            // No file has been loaded → show message and return to main menu.
            printf("\n[X] No files available for toxic analysis. "
                "Please load files in Stage 1 first.\n");
            return;
        }

        const char* activePath = (g_use_file == 1)
            ? inputFilePath1
            : inputFilePath2;
        printf("\n=== TOXIC CONTENT DETECTION ===\n");
        printf("Current source file  : %s (%s)\n",
            (g_use_file == 1) ? "File 1" : "File 2",
            activePath);
        printf("--------------------------------\n");
        printf("1. Toxic Analysis\n");
        printf("2. Dictionary Management\n");
        printf("0. Back\n");
        printf("Select: ");

        if (scanf("%d", &sub) != 1) {
            int c;
            while ((c = getchar()) != '\n' && c != EOF);
            sub = -1;
            continue;
        }

        int c;
        while ((c = getchar()) != '\n' && c != EOF);

        switch (sub) {
        case 1: {
            // (1) Require that at least one file has been loaded.
            if (!auto_select_source_file()) {
                printf("[X] No files available. Please load files in Stage 1 first.\n");
                break;
            }

            const char* activePath = (g_use_file == 1) ? inputFilePath1 : inputFilePath2;

            // (2) Require that Word Analysis has been run for this file.
            if (!analysis_data.text_filtered ||
                strcmp(current_filename, activePath) != 0) {

                printf("\n[X] Word Analysis not found for this file.\n");
                printf("Go to Menu 2: Word Analysis first.\n\n");
                break;
            }

            printf("Running toxic analysis on %s: %s\n",
                (g_use_file == 1) ? "File 1" : "File 2",
                activePath);

            toxic_analysis();
            break;
        }
        case 2:
            dictionary_management();
            break;
        case 0:
            printf("Returning to main menu...\n");
            break;
        default:
            printf("Invalid option. Please enter 0-2.\n");
        }
    } while (sub != 0);
}


// ===== 4. Stage 4 - Core Pair Building and Sorting Algorithms ======

// Compare two Pair items by the primary key only (frequency or alphabet).
static int cmp_primary(const Pair* a, const Pair* b, SortKey key) {
    if (key == KEY_FREQ_DESC) {
        if (a->count != b->count) return (b->count - a->count); // Sort by frequency in descending order.
        return 0; // If frequencies are equal, primary key does not decide the order.
    }
    else { // KEY_ALPHA
        int s = strcmp(a->word, b->word);
        if (s != 0) return s;
        return 0; // If words are identical, primary key does not decide the order.
    }
}

// Compare two Pair items using primary key and optional secondary tiebreak.
static int cmp_pairs(const Pair* a, const Pair* b, SortKey key) {
    int c = cmp_primary(a, b, key);
    if (c != 0 || !g_use_secondary_tiebreak) return c;

    // Apply the secondary key when primary key is tied.
    if (key == KEY_FREQ_DESC) {
        // When frequencies are equal, fall back to alphabetical order.
        return strcmp(a->word, b->word);
    }
    else { // KEY_ALPHA
        // When words are equal, fall back to frequency (higher frequency first).
        return (b->count - a->count);
    }
}

// Sort a Pair array using Bubble Sort, tracking statistics.
static void bubble_sort_pairs(Pair a[], int n, SortKey key) {
    for (int i = 0; i < n - 1; ++i) {
        int swapped = 0;
        for (int j = 0; j < n - 1 - i; ++j) {
            if (cmp_with_stats(&a[j], &a[j + 1], key) > 0) {
                swap_pair(&a[j], &a[j + 1]);
                swapped = 1;
            }
        }
        if (!swapped) break;
    }
}

// Partition function for Quick Sort on Pair arrays, tracking moves and comparisons.
static int partition_pairs(Pair a[], int l, int r, SortKey key) {
    Pair pivot = a[r];               g_stats.moves++; // Reading pivot into a local variable counts as one move.
    int i = l - 1;
    for (int j = l; j < r; ++j) {
        if (cmp_with_stats(&a[j], &pivot, key) <= 0) {
            ++i; swap_pair(&a[i], &a[j]);
        }
    }
    swap_pair(&a[i + 1], &a[r]);
    return i + 1;
}

// Recursive Quick Sort for Pair arrays using the common comparator.
static void quick_sort_pairs(Pair a[], int l, int r, SortKey key) {
    if (l >= r) return;
    int p = partition_pairs(a, l, r, key);
    quick_sort_pairs(a, l, p - 1, key);
    quick_sort_pairs(a, p + 1, r, key);
}

// Merge step for Merge Sort on Pair arrays, tracking data moves.
static void merge_pairs(Pair a[], int l, int m, int r, SortKey key) {
    int n1 = m - l + 1, n2 = r - m;
    Pair* L = (Pair*)malloc(sizeof(Pair) * n1);
    Pair* R = (Pair*)malloc(sizeof(Pair) * n2);
    for (int i = 0; i < n1; ++i) { L[i] = a[l + i]; g_stats.moves++; }
    for (int j = 0; j < n2; ++j) { R[j] = a[m + 1 + j]; g_stats.moves++; }

    int i = 0, j = 0, k = l;
    while (i < n1 && j < n2) {
        if (cmp_with_stats(&L[i], &R[j], key) <= 0) { a[k++] = L[i++]; g_stats.moves++; }
        else { a[k++] = R[j++]; g_stats.moves++; }
    }
    while (i < n1) { a[k++] = L[i++]; g_stats.moves++; }
    while (j < n2) { a[k++] = R[j++]; g_stats.moves++; }
    free(L); free(R);
}

// Recursive Merge Sort for Pair arrays using the common comparator.
static void merge_sort_pairs(Pair a[], int l, int r, SortKey key) {
    if (l >= r) return;
    int m = (l + r) / 2;
    merge_sort_pairs(a, l, m, key);
    merge_sort_pairs(a, m + 1, r, key);
    merge_pairs(a, l, m, r, key);
}

// Dispatch to the selected sorting algorithm and measure elapsed time.
static void sort_pairs(Pair a[], int n, SortKey key, SortAlg alg) {
    if (n <= 1) return;
    double t0 = now_ms();
    switch (alg) {
    case ALG_BUBBLE: bubble_sort_pairs(a, n, key); break;
    case ALG_QUICK:  quick_sort_pairs(a, 0, n - 1, key); break;
    case ALG_MERGE:  merge_sort_pairs(a, 0, n - 1, key); break;
    default:         quick_sort_pairs(a, 0, n - 1, key); break;
    }
    g_stats.ms += (now_ms() - t0);
}

// Build an array of unique word–count pairs from a flat token list.
static int build_pairs_from_tokens(char (*words)[50], int wordCount, Pair out[], int maxOut) {
    int ucnt = 0;
    for (int i = 0; i < wordCount; ++i) {
        int k = -1;
        for (int j = 0; j < ucnt; ++j) {
            if (strcmp(words[i], out[j].word) == 0) { k = j; break; }
        }
        if (k == -1) {
            if (ucnt >= maxOut) break;
            strcpy(out[ucnt].word, words[i]);
            out[ucnt].count = 1;
            ++ucnt;
        }
        else {
            out[k].count++;
        }
    }
    return ucnt;
}



// ===== 5. Stage4 - Reporting: Top N, Toxic, Comparison, Summary, Alphabetical List ======

// Show Top N words (all tokens) using the chosen key and algorithm.
void sort_and_show_topN_all(SortKey key, SortAlg alg, int topN) {
    if (!file1Loaded && !file2Loaded) { printf("[!] No text loaded.\n"); return; }

    char (*w)[50]; int wc; pick_tokens(&w, &wc);
    if (!w || wc == 0) { printf("[!] No text loaded.\n"); return; }

    Pair* arr = (Pair*)malloc(sizeof(Pair) * 6000);
    if (!arr) { printf("[!] OOM\n"); return; }
    int n = build_pairs_from_tokens(w, wc, arr, 6000);

    sort_pairs(arr, n, key, alg);

    if (topN > n) topN = n;
    printf("\n-- Top %d (%s, %s) --\n",
        topN,
        key == KEY_FREQ_DESC ? "freq desc" : "A->Z",
        alg == ALG_BUBBLE ? "Bubble" : (alg == ALG_QUICK ? "Quick" : "Merge"));
    for (int i = 0; i < topN; ++i) {
        printf("%2d. %-20s %d\n", i + 1, arr[i].word, arr[i].count);
    }
    free(arr);
}

// Show Top N toxic words only, sorted by frequency using the chosen algorithm.
void sort_and_show_topN_toxic(SortAlg alg, int topN) {
    if (!file1Loaded && !file2Loaded) { printf("[!] No text loaded.\n"); return; }
    load_toxicwords();

    char (*w)[50]; int wc; pick_tokens(&w, &wc);
    if (!w || wc == 0) { printf("[!] No text loaded.\n"); return; }

    // First build the full frequency list.
    Pair* all = (Pair*)malloc(sizeof(Pair) * 6000);
    Pair* tox = (Pair*)malloc(sizeof(Pair) * 6000);
    if (!all || !tox) { printf("[!] OOM\n"); free(all); free(tox); return; }

    int nAll = build_pairs_from_tokens(w, wc, all, 6000);

    // Extract only the toxic words.
    int nT = 0;
    for (int i = 0; i < nAll; ++i) {
        if (is_toxic_word(all[i].word)) tox[nT++] = all[i];
    }

    if (nT == 0) { printf("[i] No toxic words found.\n"); free(all); free(tox); return; }

    // Toxic words are typically sorted by descending frequency.
    sort_pairs(tox, nT, KEY_FREQ_DESC, alg);
    if (topN > nT) topN = nT;

    printf("\n-- Toxic Top %d (freq desc, %s) --\n",
        topN, alg == ALG_BUBBLE ? "Bubble" : (alg == ALG_QUICK ? "Quick" : "Merge"));
    for (int i = 0; i < topN; ++i) {
        printf("%2d. %-20s %d\n", i + 1, tox[i].word, tox[i].count);
    }

    free(all); free(tox);
}

// Compare Bubble, Quick, and Merge sort outputs and performance on Top N results.
void compare_algorithms_topN(int topN) {
    if (!file1Loaded && !file2Loaded) {
        printf("[!] No text loaded.\n");
        return;
    }
    char (*w)[50]; int wc;
    pick_tokens(&w, &wc);
    if (!w || wc == 0) {
        printf("[!] No text loaded.\n");
        return;
    }

    Pair* base = (Pair*)malloc(sizeof(Pair) * 6000);
    Pair* a = (Pair*)malloc(sizeof(Pair) * 6000);  // Bubble
    Pair* b = (Pair*)malloc(sizeof(Pair) * 6000);  // Quick
    Pair* c = (Pair*)malloc(sizeof(Pair) * 6000);  // Merge
    if (!base || !a || !b || !c) {
        printf("[!] OOM\n");
        free(base); free(a); free(b); free(c);
        return;
    }

    int n = build_pairs_from_tokens(w, wc, base, 6000);
    memcpy(a, base, sizeof(Pair) * n);
    memcpy(b, base, sizeof(Pair) * n);
    memcpy(c, base, sizeof(Pair) * n);

    SortKey key = g_key;

    // Measure statistics for each algorithm independently.
    SortStats sB, sQ, sM;

    stats_reset(); sort_pairs(a, n, key, ALG_BUBBLE); sB = g_stats;
    stats_reset(); sort_pairs(b, n, key, ALG_QUICK);  sQ = g_stats;
    stats_reset(); sort_pairs(c, n, key, ALG_MERGE);  sM = g_stats;

    if (topN > n) topN = n;

    printf("\n== Algorithm Comparison (key=%s, tiebreak=%s) ==\n",
        key == KEY_FREQ_DESC ? "freq desc" : "A→Z",
        g_use_secondary_tiebreak ? "ON" : "OFF");

    // Print three columns side by side: Bubble / Quick / Merge.
    printf("\n%-4s | %-20s %6s | %-20s %6s | %-20s %6s\n",
        "Rank",
        "Bubble", "count",
        "Quick", "count",
        "Merge", "count");
    printf("------+---------------------------+---------------------------+---------------------------\n");

    for (int i = 0; i < topN; i++) {
        printf("%-4d | %-20s %6d | %-20s %6d | %-20s %6d\n",
            i + 1,
            a[i].word, a[i].count,
            b[i].word, b[i].count,
            c[i].word, c[i].count);
    }

    // Stability check: compare whether the first `cap` entries are identical across algorithms.
    int agree = 1, cap = topN < 30 ? topN : 30;
    for (int i = 0; i < cap; i++) {
        if (strcmp(a[i].word, b[i].word) != 0 ||
            strcmp(a[i].word, c[i].word) != 0) {
            agree = 0;
            break;
        }
    }

    printf("\n[Stability] Top %d %s across algorithms.\n",
        cap, agree ? "ARE IDENTICAL" : "DIFFER");

    //  Performance comparison table (keeps the original formatting).
    printf("\n%-8s | %10s | %12s | %12s\n",
        "Alg", "Time(ms)", "Comparisons", "Moves");
    printf("----------+------------+--------------+--------------\n");
    printf("%-8s | %10.3f | %12lld | %12lld\n",
        "Bubble", sB.ms, sB.comps, sB.moves);
    printf("%-8s | %10.3f | %12lld | %12lld\n",
        "Quick", sQ.ms, sQ.comps, sQ.moves);
    printf("%-8s | %10.3f | %12lld | %12lld\n",
        "Merge", sM.ms, sM.comps, sM.moves);

    free(base); free(a); free(b); free(c);
}

// Print extra summary statistics such as toxic vs non-toxic ratios.
void show_extra_summary(void) {
    if (!file1Loaded && !file2Loaded) { printf("[!] No text loaded.\n"); return; }
    load_toxicwords();

    char (*w)[50]; int wc; pick_tokens(&w, &wc);
    if (!w || wc == 0) { printf("[!] No text loaded.\n"); return; }

    int toxic_tokens = 0, nontoxic_tokens = 0;
    Pair* all = (Pair*)malloc(sizeof(Pair) * 6000);
    if (!all) { printf("[!] OOM\n"); return; }
    int n = build_pairs_from_tokens(w, wc, all, 6000);

    for (int i = 0; i < n; i++) {
        if (is_toxic_word(all[i].word)) toxic_tokens += all[i].count;
        else                             nontoxic_tokens += all[i].count;
    }
    int total = toxic_tokens + nontoxic_tokens;
    double tox_ratio = total ? (100.0 * toxic_tokens / total) : 0.0;

    // Also compute ratios at the "type" level (unique words).
    int toxic_types = 0, nontoxic_types = 0;
    for (int i = 0; i < n; i++) {
        if (is_toxic_word(all[i].word)) toxic_types++;
        else                            nontoxic_types++;
    }

    printf("\n=== Extra Summary ===\n");
    printf("Tokens  : toxic=%d, non-toxic=%d, total=%d, toxic ratio=%.2f%%\n",
        toxic_tokens, nontoxic_tokens, total, tox_ratio);
    printf("Types   : toxic=%d, non-toxic=%d, total=%d\n",
        toxic_types, nontoxic_types, toxic_types + nontoxic_types);

    free(all);
}

// List all unique words alphabetically with pagination.
void list_alpha_all(void) {
    if (!file1Loaded && !file2Loaded) {
        printf("[!] No text loaded.\n");
        return;
    }

    char (*w)[50];
    int wc;
    pick_tokens(&w, &wc);
    if (!w || wc == 0) {
        printf("[!] No text loaded.\n");
        return;
    }

    Pair* arr = (Pair*)malloc(sizeof(Pair) * 6000);
    if (!arr) {
        printf("[!] OOM\n");
        return;
    }

    int n = build_pairs_from_tokens(w, wc, arr, 6000);
    // Sort the unique words alphabetically.
    sort_pairs(arr, n, KEY_ALPHA, g_alg);

    const int perPage = 50;
    int page = 0;
    char cmd[16];

    for (;;) {
        int start = page * perPage;
        if (start >= n) {
            printf("[i] No more words.\n");
            break;
        }
        int end = start + perPage;
        if (end > n) end = n;

        printf("\n-- Alphabetical listing (words %d-%d of %d) --\n",
            start + 1, end, n);
        for (int i = start; i < end; ++i) {
            printf("%-20s %d\n", arr[i].word, arr[i].count);
        }

        // Prompt the user for navigation commands.
        if (page == 0 && end == n) {
            printf("[Q]uit : ");
        }
        else if (page == 0) {
            printf("[N]ext, [Q]uit : ");
        }
        else if (end == n) {
            printf("[P]rev, [Q]uit : ");
        }
        else {
            printf("[P]rev, [N]ext, [Q]uit : ");
        }

        if (!read_line(cmd, sizeof(cmd))) {
            break;
        }
        // Empty input: do nothing and re-prompt.
        if (cmd[0] == '\0') {
            continue;
        }

        // Strictly parse: only accept a single character n / p / q (case-insensitive).
        if (strcmp(cmd, "n") == 0 || strcmp(cmd, "N") == 0) {
            if ((page + 1) * perPage >= n) {
                printf("[i] Already at last page.\n");
            }
            else {
                page++;
            }
        }
        else if (strcmp(cmd, "p") == 0 || strcmp(cmd, "P") == 0) {
            if (page == 0) {
                printf("[i] Already at first page.\n");
            }
            else {
                page--;
            }
        }
        else if (strcmp(cmd, "q") == 0 || strcmp(cmd, "Q") == 0 || strcmp(cmd, "0") == 0) {
            break;
        }
        else {
            // Inputs like "npq", "nn", or "x" will fall into this branch.
            printf("[i] Unknown command. Please use n / p / q.\n");
        }
    }
    free(arr);
}



// ===== 6. Stage 4/5 - Saving Reports (TXT + optional CSV)

// Write a full analysis report for the given token list into the provided FILE*.
static void write_full_report(FILE* f,
    const char* sourcePath,
    char (*words)[50],
    int wordCount)
{
    // Load toxic word dictionary (for basic toxicity analysis in this report).
    load_toxicwords();

    // ===== 1. Compute unique words and their frequencies (basic statistics) =====
    char uniq[1000][50];
    int  freq[5000];
    int  ucnt = 0;

    for (int i = 0; i < wordCount && ucnt < 1000; i++) {
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

    // Sort unique words by frequency (descending) and then alphabetically (A–Z).
    for (int i = 0; i < ucnt - 1; i++) {
        for (int j = 0; j < ucnt - i - 1; j++) {
            if (freq[j] < freq[j + 1] ||
                (freq[j] == freq[j + 1] && strcmp(uniq[j], uniq[j + 1]) > 0)) {

                int tf = freq[j];
                freq[j] = freq[j + 1];
                freq[j + 1] = tf;

                char tmp[50];
                strcpy(tmp, uniq[j]);
                strcpy(uniq[j], uniq[j + 1]);
                strcpy(uniq[j + 1], tmp);
            }
        }
    }

    // ===== 2. Basic toxicity analysis (based on uniq/freq + is_toxic_word) =====
    int toxic_words_count_basic = 0;        // Unique toxic words detected by basic check.
    int total_toxic_occurrences_basic = 0;  // Total occurrences of those toxic words.

    // To keep the report readable, only keep the top 100 toxic words for printing.
    char toxic_words_list[100][50];
    int  toxic_freq[100] = { 0 };
    int  toxic_severity[100] = { 0 };

    for (int i = 0; i < ucnt; i++) {
        if (is_toxic_word(uniq[i])) {
            if (toxic_words_count_basic < 100) {
                strcpy(toxic_words_list[toxic_words_count_basic], uniq[i]);
                toxic_freq[toxic_words_count_basic] = freq[i];
                toxic_severity[toxic_words_count_basic] = get_toxic_severity(uniq[i]);
                toxic_words_count_basic++;
            }
            total_toxic_occurrences_basic += freq[i];
        }
    }

    // Sort the kept toxic words by frequency (descending).
    for (int i = 0; i < toxic_words_count_basic - 1; i++) {
        for (int j = 0; j < toxic_words_count_basic - i - 1; j++) {
            if (toxic_freq[j] < toxic_freq[j + 1]) {
                int tf = toxic_freq[j];
                toxic_freq[j] = toxic_freq[j + 1];
                toxic_freq[j + 1] = tf;

                char tmpw[50];
                strcpy(tmpw, toxic_words_list[j]);
                strcpy(toxic_words_list[j], toxic_words_list[j + 1]);
                strcpy(toxic_words_list[j + 1], tmpw);

                int ts = toxic_severity[j];
                toxic_severity[j] = toxic_severity[j + 1];
                toxic_severity[j + 1] = ts;
            }
        }
    }

    // ===== 3. Detect which advanced features have data in this session =====
    // Advanced text statistics: only valid if they refer to this same source file.
    int has_advanced_text_stats =
        analysis_data.text_filtered &&
        current_filename[0] != '\0' &&
        (strcmp(current_filename, sourcePath) == 0);

    // Advanced toxic statistics: check if Stage 3 counters were updated.
    int has_advanced_toxic_stats =
        (analysis_data.toxic_words_count > 0 ||
            analysis_data.toxic_phrases_count > 0 ||
            analysis_data.toxicity_density > 0.0f);

    // Sorting performance: comparisons/moves/time > 0 means a sort was executed.
    int has_sort_performance =
        (g_stats.comps > 0 || g_stats.moves > 0 || g_stats.ms > 0.0);

    // ===== 4. Report header and overall summary =====
    fprintf(f, "Text Analysis Report\n");
    fprintf(f, "====================\n\n");

    // 4.1 Global overview (basic statistics always present).
    fprintf(f, "Overall Summary\n");
    fprintf(f, "Metric,Value\n");
    fprintf(f, "Source file,%s\n", sourcePath);
    fprintf(f, "File type,%s\n",
        isCSVFile(sourcePath) ? "CSV" : "Text");
    fprintf(f, "Analysis date,%s\n", __DATE__);
    fprintf(f, "Total tokens (raw input),%d\n", wordCount);
    fprintf(f, "Unique words (recomputed),%d\n", ucnt);
    fprintf(f, "Unique toxic words (basic check),%d\n", toxic_words_count_basic);
    fprintf(f, "Total toxic occurrences (basic check),%d\n",
        total_toxic_occurrences_basic);

    float toxicity_percentage_basic = 0.0f;
    if (wordCount > 0) {
        toxicity_percentage_basic =
            (float)total_toxic_occurrences_basic / wordCount * 100.0f;
    }
    fprintf(f, "Toxicity percentage (basic check),%.2f%%\n",
        toxicity_percentage_basic);
    fprintf(f, "Toxic dictionary terms loaded,%d\n", g_toxic_count);
    fprintf(f, "\n");

    // 4.2 Feature run summary – explicitly tell which analyses were run or not.
    fprintf(f, "Feature Run Summary\n");
    fprintf(f, "Feature,Status,Notes\n");
    fprintf(f,
        "Basic token statistics,GENERATED,"
        "Always computed when saving this report\n");
    fprintf(f,
        "Advanced text statistics,%s,%s\n",
        has_advanced_text_stats ? "AVAILABLE" : "NOT AVAILABLE",
        has_advanced_text_stats
        ? "Results taken from previous advanced analysis"
        : "Advanced analysis menu was not used for this file");
    fprintf(f,
        "Advanced toxic detection,%s,%s\n",
        has_advanced_toxic_stats ? "AVAILABLE" : "NOT AVAILABLE",
        has_advanced_toxic_stats
        ? "Uses dictionary-based word/phrase analysis"
        : "Dictionary-based toxic analysis was not performed or found no data");
    fprintf(f,
        "Sorting & performance,%s,%s\n",
        has_sort_performance ? "AVAILABLE" : "NOT AVAILABLE",
        has_sort_performance
        ? "Statistics captured from the last sorting operation"
        : "No sorting experiment was recorded in this session");
    fprintf(f, "\n");

    // ===== 5. Basic word-frequency statistics (always available) =====
    fprintf(f, "Basic Token Statistics\n");
    fprintf(f, "Metric,Value\n");

    int singleOccurrence = 0;
    for (int i = 0; i < ucnt; i++) {
        if (freq[i] == 1) singleOccurrence++;
    }
    float avgFrequency =
        (ucnt > 0) ? (float)wordCount / (float)ucnt : 0.0f;

    fprintf(f, "Unique word count,%d\n", ucnt);
    fprintf(f, "Words with single occurrence,%d\n", singleOccurrence);
    fprintf(f, "Average frequency per unique word,%.2f\n", avgFrequency);
    fprintf(f, "\n");

    // Top 20 most frequent words.
    fprintf(f, "Top Words by Frequency (up to 20)\n");
    fprintf(f, "Rank,Word,Frequency,%% of all tokens,Is toxic\n");
    int totalWords = wordCount;
    int topn = (ucnt < 20) ? ucnt : 20;
    for (int i = 0; i < topn; i++) {
        float percentage = (float)freq[i] / totalWords * 100.0f;
        const char* is_toxic_flag = is_toxic_word(uniq[i]) ? "Yes" : "No";
        fprintf(f, "%d,%s,%d,%.2f%%,%s\n",
            i + 1, uniq[i], freq[i], percentage, is_toxic_flag);
    }
    fprintf(f, "\n");

    // ===== 6. Advanced text statistics (corresponds to word_analysis results) =====
    fprintf(f, "Advanced Text Statistics (stopwords & normalisation)\n");
    if (has_advanced_text_stats) {
        fprintf(f, "Metric,Value\n");
        fprintf(f, "Filtered words (after stopwords),%d\n",
            analysis_data.total_words_filtered);
        fprintf(f, "Unique words (filtered),%d\n",
            analysis_data.word_count);
        fprintf(f, "Detected sentences,%d\n",
            analysis_data.sentences);

        double avgSentenceLen =
            (analysis_data.sentences > 0)
            ? (double)analysis_data.total_words_filtered /
            analysis_data.sentences
            : 0.0;
        fprintf(f, "Average sentence length (words),%.2f\n",
            avgSentenceLen);

        fprintf(f, "Stopwords filtered out,%d\n",
            analysis_data.stopwords_removed);

        double lex_div = 0.0;
        if (analysis_data.total_words_filtered > 0) {
            lex_div = (double)analysis_data.word_count /
                analysis_data.total_words_filtered;
        }
        fprintf(f, "Lexical diversity (filtered),%.3f\n", lex_div);
        fprintf(f, "Text normalisation,%s\n",
            analysis_data.variant_processing_enabled
            ? "ENABLED"
            : "DISABLED");
    }
    else {
        fprintf(f,
            "No advanced text statistics are available for this file.\n"
            "Use the advanced analysis menu before saving the report if you\n"
            "want these values to be included.\n");
    }
    fprintf(f, "\n");

    // ===== 7. Toxic content overview =====
    fprintf(f, "Toxic Content Overview\n");

    // 7.1 Basic toxic word check (always available in this report).
    fprintf(f, "Basic dictionary check (current report)\n");
    fprintf(f, "Metric,Value\n");
    fprintf(f, "Unique toxic words (basic),%d\n",
        toxic_words_count_basic);
    fprintf(f, "Total toxic occurrences (basic),%d\n",
        total_toxic_occurrences_basic);
    fprintf(f, "Toxicity percentage (basic),%.2f%%\n",
        toxicity_percentage_basic);

    // Top 10 toxic words (basic view).
    if (toxic_words_count_basic > 0) {
        fprintf(f, "Top toxic words (up to 10)\n");
        fprintf(f, "Rank,Word,Frequency,Severity,%% of all tokens\n");
        int topT = (toxic_words_count_basic < 10)
            ? toxic_words_count_basic
            : 10;
        for (int i = 0; i < topT; i++) {
            float wp = (float)toxic_freq[i] / wordCount * 100.0f;
            fprintf(f, "%d,%s,%d,%d,%.2f%%\n",
                i + 1,
                toxic_words_list[i],
                toxic_freq[i],
                toxic_severity[i],
                wp);
        }
    }
    else {
        fprintf(f,
            "The basic dictionary check did not find any toxic words "
            "in this file.\n");
    }
    fprintf(f, "\n");

    // 7.2 Extended dictionary/phrase analysis (if Stage 3 data exist).
    fprintf(f, "Dictionary & phrase-based toxic analysis (if available)\n");
    if (has_advanced_toxic_stats) {
        fprintf(f, "Metric,Value\n");
        fprintf(f, "Toxic words in internal list,%d\n",
            analysis_data.toxic_words_count);
        fprintf(f, "Toxic phrases (bigrams/trigrams),%d\n",
            analysis_data.toxic_phrases_count);
        fprintf(f, "Total toxic occurrences (internal counters),%d\n",
            analysis_data.total_toxic_occurrences);
        fprintf(f, "Toxicity density (internal),%.4f\n",
            analysis_data.toxicity_density);
        fprintf(f, "Bigram toxic occurrences,%d\n",
            analysis_data.bigram_toxic_occurrences);
        fprintf(f, "Trigram toxic occurrences,%d\n",
            analysis_data.trigram_toxic_occurrences);
    }
    else {
        fprintf(f,
            "No extended toxic analysis data are available for this file.\n"
            "Run the toxic content detection features before saving the\n"
            "report if you want these details.\n");
    }
    fprintf(f, "\n");

    // ===== 8. Sorting / performance overview =====
    fprintf(f, "Sorting & Performance Overview\n");
    if (has_sort_performance) {
        fprintf(f, "Metric,Value\n");
        fprintf(f, "Current sort key,%s\n",
            (g_key == KEY_FREQ_DESC)
            ? "Frequency (descending, ties→alpha)"
            : "Alphabetical (A→Z)");
        fprintf(f, "Current sort algorithm,%s\n",
            (g_alg == ALG_BUBBLE)
            ? "Bubble"
            : (g_alg == ALG_QUICK ? "Quick" : "Merge"));
        fprintf(f, "Secondary tiebreak,%s\n",
            g_use_secondary_tiebreak
            ? "ON (alpha as secondary key)"
            : "OFF (pure primary key)");
        fprintf(f, "Configured Top N,%d\n", g_topN);
        fprintf(f, "Last sort comparisons,%lld\n", g_stats.comps);
        fprintf(f, "Last sort moves,%lld\n", g_stats.moves);
        fprintf(f, "Last sort time (ms),%.3f\n", g_stats.ms);
    }
    else {
        fprintf(f,
            "No sorting statistics were recorded in this session.\n"
            "Use the sorting/reporting menu before saving the report\n"
            "if you want performance numbers to appear here.\n");
    }
}

// Save the current analysis results to a TXT report and optionally a CSV report.
void saveResultsToFile(void) {
    if (!file1Loaded && !file2Loaded) {
        printf("[!] No text loaded. Use menu 1 first.\n");
        return;
    }

    // Automatically select a valid active source (File 1 / File 2) based on current state.
    if (!auto_select_source_file()) {
        printf("[X] No files available. Please load files in Stage 1 first.\n");
        return;
    }

    const char* sourcePath;
    char (*words)[50];
    int  wordCount;

    // Select the path using the current g_use_file 
    sourcePath = (g_use_file == 1) ? inputFilePath1 : inputFilePath2;

    // Use pick_tokens，ensure that it have tokens which is same with Stage 4
    pick_tokens(&words, &wordCount);

    if (!words || wordCount <= 0) {
        printf("[X] No tokens available from current source file.\n");
        printf("Tip: Make sure you have loaded the file in Stage 1.\n");
        return;
    }


    // 2. Clean up the filename entered by the user.
    char base[512];
    strncpy(base, outputFilePath, sizeof(base) - 1);
    base[sizeof(base) - 1] = '\0';
    strip_quotes(base);

    // If the user did not provide an extension, default to .txt.
    char* dot = strrchr(base, '.');
    if (dot == NULL) {
        strcat(base, ".txt");
    }

    // At this point, `base` is guaranteed to be of the form name.xxx.
    // First build txt_path.
    char txt_path[512];
    strncpy(txt_path, base, sizeof(txt_path) - 1);
    txt_path[sizeof(txt_path) - 1] = '\0';

    // Then derive csv_path from txt_path by changing the extension to .csv.
    char csv_path[512];
    strncpy(csv_path, txt_path, sizeof(csv_path) - 1);
    csv_path[sizeof(csv_path) - 1] = '\0';

    char* dot2 = strrchr(csv_path, '.');
    if (dot2 != NULL) {
        strcpy(dot2, ".csv");
    }
    else {
        strcat(csv_path, ".csv");
    }

    // ===== 3. Generate the TXT report first. =====
#ifdef _WIN32
    FILE* f_txt = fopen_u8(txt_path, "w");
#else
    FILE* f_txt = fopen(txt_path, "w");
#endif
    if (!f_txt) {
        printf("[!] Error: Cannot create output file '%s'\n", txt_path);
        perror("Detailed error");
        handleError("Cannot open text output file");
        return;
    }

    write_full_report(f_txt, sourcePath, words, wordCount);
    fclose(f_txt);
    printf("\nSaved TEXT report to %s\n", txt_path);

    // ===== 4. Ask the user whether a CSV version is also needed. =====
    printf("Do you also want a CSV version for spreadsheets? (y/n): ");
    char ans[16];
    if (!read_line(ans, sizeof(ans))) {
        // If input cannot be read, treat it as "no".
        return;
    }

    if (ans[0] != 'y' && ans[0] != 'Y') {
        printf("CSV report not generated.\n");
        return;
    }

    // ===== 5. Generate the CSV report. =====
#ifdef _WIN32
    FILE* f_csv = fopen_u8(csv_path, "w");
#else
    FILE* f_csv = fopen(csv_path, "w");
#endif
    if (!f_csv) {
        printf("[!] Error: Cannot create CSV file '%s'\n", csv_path);
        perror("Detailed error");
        handleError("Cannot open CSV output file");
        return;
    }

    write_full_report(f_csv, sourcePath, words, wordCount);
    fclose(f_csv);
    printf("Saved CSV report to %s\n", csv_path);

    // Console output summary
    printf("Your reports include:\n");
    printf(" - TEXT report: %s\n", txt_path);
    printf(" - CSV  report: %s\n", csv_path);
}


// ===== 7. Stage 4/6 - Sorting & Reporting Main Menu (User Interaction)

// Show the sorting & reporting menu and dispatch user commands.
void menu_sort_and_report(void) {
    int sub = -1;
    if (!auto_select_source_file()) {
        // No files available → show a message and return directly to the main menu.
        printf("\n[X] No files available for sorting/reporting. "
            "Please load files in Stage 1 first.\n");
        return;
    }
    do {
        // On each loop, confirm the current source (in case Stage 1 changed the active file).
        if (!auto_select_source_file()) {
            printf("\n[X] No files available.\n");
            return;
        }

        const char* activePath = (g_use_file == 1)
            ? inputFilePath1
            : inputFilePath2;
        printf("\n=== Sorting & Reporting ===\n");
        printf("Current source file      : %s (%s)\n",
            (g_use_file == 1) ? "File 1" : "File 2",
            activePath);
        printf("----------------------------------------\n");
        printf("1. Set sort KEY (current: %s)\n",
            g_key == KEY_FREQ_DESC ? "Frequency in descending" : "Alphabetical (A->Z)");
        printf("2. Set sort ALGORITHM (current: %s)\n",
            g_alg == ALG_BUBBLE ? "Bubble" : (g_alg == ALG_QUICK ? "Quick" : "Merge"));
        printf("3. Toggle secondary tiebreak (current: %s)\n",
            g_use_secondary_tiebreak ? "ON (alpha as tiebreak)" : "OFF (pure primary key)");
        printf("4. Set Top N (current: %d)\n", g_topN);
        printf("5. Show Top N (ALL words)\n");
        printf("6. Show Top N (TOXIC words only)\n");
        printf("7. Compare algorithms (Top N)\n");
        printf("8. Extra summary (toxic ratio)\n");
        printf("9  List ALL words alphabetically\n");
        printf("0. Back\n");
        printf("Select: ");

        if (scanf("%d", &sub) != 1) {
            int c;
            while ((c = getchar()) != '\n' && c != EOF);
            printf("Invalid input. Please enter 0-9.\n");
            sub = -1;
            continue;
        }

        int c;
        while ((c = getchar()) != '\n' && c != EOF);


        switch (sub) {
        case 1: {
            // Configure sorting key (frequency or alphabetical).
            int k;
            printf("Choose key: 1=Frequency desc, 2=Alphabetical : ");
            if (scanf("%d", &k) == 1) {
                g_key = (k == 2 ? KEY_ALPHA : KEY_FREQ_DESC);
            }
            else {
                printf("Invalid key choice.\n");
            }
            while ((c = getchar()) != '\n' && c != EOF);
        } break;
        case 2: {
            //Configure sorting algorithm (Bubble / Quick / Merge).
            int a;
            printf("Choose algorithm: 1=Bubble  2=Quick  3=Merge : ");
            if (scanf("%d", &a) == 1) {
                if (a == 1) g_alg = ALG_BUBBLE;
                else if (a == 3) g_alg = ALG_MERGE;
                else g_alg = ALG_QUICK;
            }
        } break;
        case 3:
            g_use_secondary_tiebreak = !g_use_secondary_tiebreak;
            printf("Secondary tiebreak is now %s\n",
                g_use_secondary_tiebreak ? "ON" : "OFF");
            break;
        case 4: {
            //Configure Top N value for reporting.
            int n;
            printf("Enter Top N (>=1): ");
            if (scanf("%d", &n) == 1 && n > 0) g_topN = n;
        } break;
        case 5:
            //Show Top N words across all tokens.
            sort_and_show_topN_all(g_key, g_alg, g_topN);
            break;
        case 6:
            //Show Top N toxic words only; typically sorted by frequency desc.
            sort_and_show_topN_toxic(g_alg, g_topN);
            break;
        case 7:
            compare_algorithms_topN(g_topN);
            break;
        case 8:
            show_extra_summary();
            break;
        case 9:
            list_alpha_all();
            break;
        case 0:
            break;
        default:
            printf("Invalid choice. Please enter 0-9.\n");
        }
    } while (sub != 0);
}

// ====== 8. Start your program ======
int main() {
    init_basic_variants();
    int userChoice;

    for (;;) {
        printf("\n=== Toxic Word Text Analyser Menu ===\n");
        printf("1. Load text files for analysis\n");
        printf("2. Advanced Text Analysis (with stopwords filtering and variants)\n");
        printf("3. Toxic Content Detection\n");
        printf("4. Sort and Display top N Words\n");
        printf("5. Save results to output file\n");
        printf("6. Exit system\n");
        printf("Enter a choice or type 6 to exit the system: ");

        if (scanf("%d", &userChoice) != 1) {
            int c; while ((c = getchar()) != '\n' && c != EOF);
            printf("Invalid input. Please enter a number 1-6!\n");
            continue;
        }
        int c; while ((c = getchar()) != '\n' && c != EOF);

        switch (userChoice) {
        case 1:
            handleFileMenu();
            break;
        case 2:
            if (!file1Loaded && !file2Loaded) {
                printf("No files loaded. Please use option 1 first to load files.\n");
            }
            else {
                display_advanced_analysis_menu();
            }
            break;
        case 3:
            if (!file1Loaded && !file2Loaded) {
                printf("No files loaded. Please use option 1 first to load files.\n");
            }
            else {
                display_toxic_menu();
            }
            break;
        case 4:
            if (!file1Loaded && !file2Loaded) {
                printf("No files loaded. Please use option 1 first to load files.\n");
            }
            else {
                menu_sort_and_report();
            }
            break;
        case 5: {
            printf("Enter output file name "
                "(press Enter for default 'analysis_report.txt'):\n> ");

            if (!read_line(outputFilePath, sizeof(outputFilePath))) {
                handleError("Failed to read output path");
                break;
            }
            if (outputFilePath[0] == '\0') {
                strcpy(outputFilePath, "analysis_report.txt");
                printf("No name entered. Using default: %s\n", outputFilePath);
            }
            saveResultsToFile();
        } break;
        case 6:
            printf("Exiting the system... Goodbye!\n");
            cleanup_analysis_data();
            return 0;
        default:
            printf("Error. %d is an invalid choice. Please enter a number between 1 and 6.\n", userChoice);
        }
    }
    cleanup_analysis_data();
    printf("\nThank you for using Text Analyser. Goodbye!\n");
    return 0;
}
