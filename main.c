#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <ctype.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>

// -------- 全局变量（更新为支持多文件） --------
char inputFilePath1[256];  // 文件1路径
char inputFilePath2[256];  // 文件2路径
char toxicWordsFile[256];  // 未使用，占位
char stopWordsFile[256];   // 未使用，占位
char outputFilePath[256];

char words1[30000][50];     // 文件1的词表
char words2[30000][50];     // 文件2的词表
char toxicWords[500][50];  // 未使用，占位

int  wordCount1 = 0;       // 文件1的词数
int  wordCount2 = 0;       // 文件2的词数
int  toxicCount = 0;       //留给stage3（还没做）

bool file1Loaded = false;  // 文件1是否已加载
bool file2Loaded = false;  // 文件2是否已加载

// ====== 新增：高级文本分析结构 ======
#define MAX_WORDS 40000
#define MAX_WORD_LENGTH 50
#define MAX_STOPWORDS 500
#define MAX_TEXT_LENGTH 50000

//高级分析时使用，记录唯一词及其频次
struct WordInfo {
    char word[MAX_WORD_LENGTH];
    int count;
};

typedef enum { KEY_FREQ_DESC, KEY_ALPHA } SortKey;  //排序“键”（频率降序 / 字母序）
typedef enum { ALG_BUBBLE, ALG_QUICK, ALG_MERGE } SortAlg;  //排序算法类型
typedef struct {
    char word[50];
    int  count;
} Pair;
typedef struct {
    long long comps;   // 比较次数
    long long moves;   // 元素移动/交换次数
    double    ms;      // 排序耗时(ms)
} SortStats;

static int cmp_pairs(const Pair* a, const Pair* b, SortKey key);
static void pick_tokens(char (**out_words)[50], int* out_count);
static SortStats g_stats;

static void stats_reset(void) { g_stats.comps = 0; g_stats.moves = 0; g_stats.ms = 0.0; }

// 计比较
static inline int cmp_with_stats(const Pair* a, const Pair* b, SortKey key) {
    g_stats.comps++;
    return cmp_pairs(a, b, key);
}

// 计移动（交换/写入）
static inline void swap_pair(Pair* x, Pair* y) {
    Pair t = *x; *x = *y; *y = t; g_stats.moves += 3; // 粗略按3次赋值算
}

static inline double now_ms(void) {
    return 1000.0 * clock() / CLOCKS_PER_SEC;
}

//全局默认排序设置与默认 Top N
static SortKey g_key = KEY_FREQ_DESC;   // 默认按频率
static SortAlg g_alg = ALG_QUICK;       // 默认快速排序
static int     g_topN = 10;             // 默认Top N
static int g_use_secondary_tiebreak = 1; // 1=使用二级键(现在的行为), 0=纯主键，制造ties观察稳定性
static int g_use_file = 1; // 1=File1, 2=File2, 0=auto(现状)

// -------- 函数声明（更新） --------
void loadTextFile(int fileNumber);
void displayGeneralStatistics();
void displayToxicWordAnalysis();
void sortAndDisplayTopNWords();
void saveResultsToFile();
void loadDictionaries();
void handleError(const char* message);
void displayNegativityScale();
void displayWordOccurrence();
void showFileHistory(int fileNumber);
void handleFileMenu();
bool isCSVFile(const char* filename);
void processCSVFile(FILE* f, char (*targetWords)[50], int* targetWordCount);
bool isFileCorrupted(const char* filePath, const char* fileContent, size_t contentSize); // New: File corruption detection
void displayFileRecoveryTips(const char* problemType);
static int  build_pairs_from_tokens(char (*words)[50], int wordCount, Pair out[], int maxOut);
static void sort_pairs(Pair a[], int n, SortKey key, SortAlg alg);
static int  load_toxicwords(void);
static int  is_toxic_word(const char* w);
void        menu_sort_and_report(void);   // 你在 main 里调用了它，提前声明
static void merge_sort_pairs(Pair a[], int l, int r, SortKey key);

// ====== 新增：高级文本分析函数声明 ======
int load_stopwords(char stopwords[][MAX_WORD_LENGTH]);
int is_stopword(char* word, char stopwords[][MAX_WORD_LENGTH], int stop_count);
int tokenize_and_clean(char* text, struct WordInfo words[], char stopwords[][MAX_WORD_LENGTH], int stop_count, int* total_chars);
int count_sentences(char* text);
void generate_statistics(struct WordInfo words[], int word_count, int total_words, int total_chars, int sentences, int stopwords_removed);
void sort_by_frequency(struct WordInfo words[], int count);
void show_top_words(struct WordInfo words[], int word_count, int n);
void advancedTextAnalysis();
void sort_and_show_topN_all(SortKey key, SortAlg alg, int topN);
void sort_and_show_topN_toxic(SortAlg alg, int topN);
void compare_algorithms_topN(int topN);          // Distinction：对比多算法TopN
void show_extra_summary(void);                   // Distinction：毒性比例等
void list_alpha_all(void);                       // 可选：按字母全部列出（分页亦可）

// ====== 新增：通用小工具（直接可用）======
// 读一整行（含空格），并去掉行尾 \n\r
static int read_line(char* buf, size_t cap) {
    if (!fgets(buf, cap, stdin)) return 0;
    size_t n = strlen(buf);
    while (n && (buf[n - 1] == '\n' || buf[n - 1] == '\r')) buf[--n] = '\0';
    return 1;
}

// 允许输入路径带引号（从资源管理器复制时常见），自动剥掉
static void strip_quotes(char* s) {
    size_t n = strlen(s);
    if (n >= 2 && s[0] == '"' && s[n - 1] == '"') {
        memmove(s, s + 1, n - 2);
        s[n - 2] = '\0';
    }
}

// 把一行规范化为：小写 + 非字母数字转空格
static void normalize_line(char* s) {
    for (char* p = s; *p; ++p) {
        unsigned char c = (unsigned char)*p;
        if (isalnum(c)) *p = (char)tolower(c);
        else *p = ' ';
    }
}

// --- Windows UTF-8 路径支持（中文/非 ASCII 更稳）---
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

// 统一的只读打开：自动去引号、Windows 用宽字符
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
        fprintf(stderr, "[!] Failed to open file for: %s\n", tmp);
        perror("reason");
    }
    return f;
}

//file corruption detection - FIXED VERSION (plsssssssssssssssssssssssssssssssssss DONT CHANGE)
bool isFileCorrupted(const char* filePath, const char* fileContent, size_t contentSize) {
    if (contentSize == 0) {
        printf("[!] Error : We don't accept empty files!\n");
        return false; // Empty file is not considered corrupted, just no content
    }

    int totalChars = 0;
    int printableChars = 0;
    int controlChars = 0;
    int weirdChars = 0;
    int consecutiveWeird = 0;
    int maxConsecutiveWeird = 0;
    int wordLikeSequences = 0;
    int binaryMarkers = 0;

    // Define what we consider "weird" characters (non-printable, non-whitespace)
    for (size_t i = 0; i < contentSize; i++) {
        unsigned char c = (unsigned char)fileContent[i];
        totalChars++;

        // Normal characters: printable ASCII + common whitespace
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

    printf("[i] File analysis: %zu chars, %.1f%% printable, %.1f%% weird, word density: %.1f/100chars\n", 
           contentSize, printableRatio * 100, weirdRatio * 100, wordDensity);

    // CORRUPTED FILE DETECTION - ONLY IF MORE THAN HALF IS WEIRD
    bool likelyCorrupted = false;
    const char* reason = NULL;

    // MAIN CONDITION: Only corrupt if more than 50% of file is weird characters
    if (weirdRatio > 0.50) {
        likelyCorrupted = true;
        reason = "more than half the file contains weird/nonsense characters";
    }
    // Additional safety checks for extreme cases
    else if (printableRatio < 0.10 && contentSize > 100) {
        likelyCorrupted = true;
        reason = "extremely low percentage of printable characters (less than 10%)";
    }
    else if (maxConsecutiveWeird > 100) {
        likelyCorrupted = true;
        reason = "extremely long sequences of consecutive weird characters";
    }
    else if (wordDensity < 0.1 && contentSize > 500 && weirdRatio > 0.30) {
        likelyCorrupted = true;
        reason = "very low word density with significant weird characters";
    }

    if (likelyCorrupted) {
        printf("\n[X] ERROR: Corrupted file detected!\n");
        printf("[X] Reason: %s\n", reason);
        printf("[X] The file contains too many nonsense characters.\n");
        printf("\nRecovery instructions:\n");
        printf("1. This file contains too many nonsense characters (>50%%)\n");
        printf("2. Please select a valid text file with mostly readable content\n");
        printf("3. Ensure the file contains proper text, not binary data\n");
        printf("4. Try opening the file in a text editor to verify its contents\n");
        return true;
    }

    // File is normal - can contain some weird characters but not too many
    printf("[✓] File is valid - contains %.1f%% weird characters (acceptable)\n", weirdRatio * 100);
    
    // Show content preview
    if (contentSize > 0) {
        printf("Content preview: ");
        int previewSize = (contentSize < 60) ? contentSize : 60;
        for (int i = 0; i < previewSize; i++) {
            unsigned char c = (unsigned char)fileContent[i];
            if (c >= 32 && c <= 126) {
                putchar(c);
            } else if (c == '\t' || c == '\n' || c == '\r') {
                printf(" ");
            } else {
                printf("?"); // Show ? for occasional weird chars
            }
        }
        printf("\n");
    }
    
    return false; // file is normal
}

// ====== 主程序（更新了选项1的处理）======
int main() {
    int userChoice;

    for (;;) {
        printf("\n\nToxic Word Text Analyser Menu\n");
        printf("1. Load text file for analysis (e.g., enter text file path)\n");
        printf("2. Display general word statistics (e.g., word counts, frequencies)\n");
        printf("3. Display toxic word analysis (e.g., toxicity score)\n");
        printf("4. Sort and display top N words (e.g., by frequency or toxicity)\n");
        printf("5. Save results to output file\n");
        printf("6. Advanced Text Analysis (with stopwords filtering)\n");
        printf("7. Exit program\n");
        printf("Type in a choice or type 8 to exit the system: ");

        if (scanf("%d", &userChoice) != 1) {
            // 清理错误输入
            int c; while ((c = getchar()) != '\n' && c != EOF);
            printf("Invalid input. Please enter a number 1-8!\n");
            continue;
        }
        // 吃掉行尾换行，方便后面用 read_line 读路径
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
            menu_sort_and_report();
            break;

        case 5:
            printf("Enter output file path (e.g., analysis_report.txt):\n> ");
            if (!read_line(outputFilePath, sizeof(outputFilePath))) {
                handleError("Failed to read output path");
                break;
            }
            saveResultsToFile();
            break;

        case 6:
            advancedTextAnalysis();
            break;

        case 7:
            printf("Exiting the system... Goodbye!\n");
            return 0;  // 真正退出

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
        printf("[!] Error: Cannot open file '%s'\n", filePath);
        printf("Recovery instructions:\n");
        printf("1. Make sure the file name is correct\n");
        printf("2. Check if the file exists in the specified location\n");
        printf("3. Move the file to the same directory as this program\n");
        printf("4. Ensure you have read permissions for the file\n");
        printf("5. If using spaces in path, enclose the entire path in quotes\n");
        printf("6. Example: \"C:/My Documents/file.txt\" or just \"filename.txt\"\n");
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

// ====== 新增：高级文本分析函数实现 ======

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

    printf("Text length: %zu characters\n", strlen(text));

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
    printf("Text buffer used: %zu/%d characters\n", strlen(text), MAX_TEXT_LENGTH);

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

// ====== 原有的CSV和文件处理函数保持不变 ======

// ====== 新增：CSV 文件支持函数 ======
// 检查文件是否为CSV格式
bool isCSVFile(const char* filename) {
    const char* dot = strrchr(filename, '.');
    if (dot != NULL) {
        return (strcmp(dot, ".csv") == 0 || strcmp(dot, ".CSV") == 0);
    }
    return false;
}

// 处理CSV文件：将列转换为文本
void processCSVFile(FILE* f, char (*targetWords)[50], int* targetWordCount) {
    char line[4096];
    int columnCount = 0;
    char* columns[100]; // 假设最多100列

    printf("[i] Processing CSV file...\n");

    while (fgets(line, sizeof(line), f) && *targetWordCount < 30000) {
        // 移除行尾换行符
        line[strcspn(line, "\n\r")] = '\0';

        // CSV解析：简单的逗号分割
        columnCount = 0;
        char* token = strtok(line, ",");
        while (token != NULL && columnCount < 100) {
            columns[columnCount++] = token;
            token = strtok(NULL, ",");
        }

        // 处理每一列
        for (int i = 0; i < columnCount && *targetWordCount < 30000; i++) {
            // 复制列内容到临时缓冲区
            char columnText[256];
            strncpy(columnText, columns[i], sizeof(columnText) - 1);
            columnText[sizeof(columnText) - 1] = '\0';

            // 规范化文本
            normalize_line(columnText);

            // 分词处理
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

// ====== 新增：文件菜单处理函数 ======
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

// ====== 新增：显示文件历史函数（显示CSV到TXT转换） ======
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

    // 显示前10个单词作为样本
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

// --- Stage 4: stub implementations to satisfy linker ---
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

void sort_and_show_topN_toxic(SortAlg alg, int topN) {
    if (!file1Loaded && !file2Loaded) { printf("[!] No text loaded.\n"); return; }
    load_toxicwords();

    char (*w)[50]; int wc; pick_tokens(&w, &wc);
    if (!w || wc == 0) { printf("[!] No text loaded.\n"); return; }

    // 先做全量词频
    Pair* all = (Pair*)malloc(sizeof(Pair) * 6000);
    Pair* tox = (Pair*)malloc(sizeof(Pair) * 6000);
    if (!all || !tox) { printf("[!] OOM\n"); free(all); free(tox); return; }

    int nAll = build_pairs_from_tokens(w, wc, all, 6000);

    // 过滤毒词
    int nT = 0;
    for (int i = 0; i < nAll; ++i) {
        if (is_toxic_word(all[i].word)) tox[nT++] = all[i];
    }

    if (nT == 0) { printf("[i] No toxic words found.\n"); free(all); free(tox); return; }

    sort_pairs(tox, nT, KEY_FREQ_DESC, alg); // 毒词一般按频率降序
    if (topN > nT) topN = nT;

    printf("\n-- Toxic Top %d (freq desc, %s) --\n",
        topN, alg == ALG_BUBBLE ? "Bubble" : (alg == ALG_QUICK ? "Quick" : "Merge"));
    for (int i = 0; i < topN; ++i) {
        printf("%2d. %-20s %d\n", i + 1, tox[i].word, tox[i].count);
    }

    free(all); free(tox);
}

void compare_algorithms_topN(int topN) {
    if (!file1Loaded && !file2Loaded) { printf("[!] No text loaded.\n"); return; }
    char (*w)[50]; int wc; pick_tokens(&w, &wc);
    if (!w || wc == 0) { printf("[!] No text loaded.\n"); return; }

    Pair* base = (Pair*)malloc(sizeof(Pair) * 6000);
    Pair* a = (Pair*)malloc(sizeof(Pair) * 6000);
    Pair* b = (Pair*)malloc(sizeof(Pair) * 6000);
    Pair* c = (Pair*)malloc(sizeof(Pair) * 6000);
    if (!base || !a || !b || !c) { printf("[!] OOM\n"); free(base); free(a); free(b); free(c); return; }

    int n = build_pairs_from_tokens(w, wc, base, 6000);
    memcpy(a, base, sizeof(Pair) * n);
    memcpy(b, base, sizeof(Pair) * n);
    memcpy(c, base, sizeof(Pair) * n);

    SortKey key = g_key;

    // ⬇️ 每种算法单独统计
    SortStats sB, sQ, sM;

    stats_reset(); sort_pairs(a, n, key, ALG_BUBBLE); sB = g_stats;
    stats_reset(); sort_pairs(b, n, key, ALG_QUICK); sQ = g_stats;
    stats_reset(); sort_pairs(c, n, key, ALG_MERGE); sM = g_stats;

    if (topN > n) topN = n;

    printf("\n== Algorithm Comparison (key=%s, tiebreak=%s) ==\n",
        key == KEY_FREQ_DESC ? "freq desc" : "A→Z",
        g_use_secondary_tiebreak ? "ON" : "OFF");

    // 结果 TopN（可见“稳定/不稳定”导致顺序差异）
    printf("-- Bubble --\n");
    for (int i = 0; i < topN; i++) printf("%2d. %-20s %d\n", i + 1, a[i].word, a[i].count);

    printf("-- Quick  --\n");
    for (int i = 0; i < topN; i++) printf("%2d. %-20s %d\n", i + 1, b[i].word, b[i].count);

    printf("-- Merge  --\n");
    for (int i = 0; i < topN; i++) printf("%2d. %-20s %d\n", i + 1, c[i].word, c[i].count);

    // 稳定性一致性检查
    int agree = 1, cap = topN < 30 ? topN : 30;
    for (int i = 0; i < cap; i++) {
        if (strcmp(a[i].word, b[i].word) != 0 || strcmp(a[i].word, c[i].word) != 0) { agree = 0; break; }
    }

    // ⬇️ 打印“性能对比表”
    printf("\n[Stability] Top %d %s across algorithms.\n", cap, agree ? "ARE IDENTICAL" : "DIFFER");
    printf("\n%-8s | %10s | %12s | %12s\n", "Alg", "Time(ms)", "Comparisons", "Moves");
    printf("----------+------------+--------------+--------------\n");
    printf("%-8s | %10.3f | %12lld | %12lld\n", "Bubble", sB.ms, sB.comps, sB.moves);
    printf("%-8s | %10.3f | %12lld | %12lld\n", "Quick", sQ.ms, sQ.comps, sQ.moves);
    printf("%-8s | %10.3f | %12lld | %12lld\n", "Merge", sM.ms, sM.comps, sM.moves);

    free(base); free(a); free(b); free(c);
}

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

    // 也可计算“类型”层面的比例（unique）
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

void list_alpha_all(void) {
    if (!file1Loaded && !file2Loaded) { printf("[!] No text loaded.\n"); return; }
    char (*w)[50]; int wc; pick_tokens(&w, &wc);
    if (!w || wc == 0) { printf("[!] No text loaded.\n"); return; }

    Pair* arr = (Pair*)malloc(sizeof(Pair) * 6000);
    if (!arr) { printf("[!] OOM\n"); return; }
    int n = build_pairs_from_tokens(w, wc, arr, 6000);
    sort_pairs(arr, n, KEY_ALPHA, g_alg); // 用当前算法做字母序

    int show = n < 200 ? n : 200;
    printf("\n-- Alphabetical listing (first %d/%d) --\n", show, n);
    for (int i = 0; i < show; i++) printf("%-20s %d\n", arr[i].word, arr[i].count);
    if (n > 200) printf("... (%d more)\n", n - 200);
    free(arr);
}


// 把 tokens -> 唯一词频 Pair[]，返回唯一词个数
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

static int g_toxic_loaded = 0;
static int g_toxic_count = 0;
static char g_toxic[500][MAX_WORD_LENGTH];

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

static int is_toxic_word(const char* w) {
    for (int i = 0; i < g_toxic_count; ++i)
        if (strcmp(w, g_toxic[i]) == 0) return 1;
    return 0;

}

void menu_sort_and_report(void) {
    int sub = -1;
    do {
        printf("\n=== Sorting & Reporting ===\n");
        printf("1) Set source file (current: %s)\n",
            g_use_file == 1 ? "File 1" : (g_use_file == 2 ? "File 2" : "Auto(File1->File2)"));
        printf("2) Set sort KEY (current: %s)\n",
            g_key == KEY_FREQ_DESC ? "Frequency desc (ties->alpha)" : "Alphabetical (A->Z)");
        printf("3) Set sort ALGORITHM (current: %s)\n",
            g_alg == ALG_BUBBLE ? "Bubble" : (g_alg == ALG_QUICK ? "Quick" : "Merge"));
        printf("4) Toggle secondary tiebreak (current: %s)\n",
            g_use_secondary_tiebreak ? "ON (alpha as tiebreak)" : "OFF (pure primary key)");
        printf("5) Set Top N (current: %d)\n", g_topN);
        printf("6) Show Top N (ALL words)\n");           // Pass: Top N 高频词
        printf("7) Show Top N (TOXIC words only)\n");    // Merit: 单独Top N毒词
        printf("8) Compare algorithms (Top N)\n");       // Distinction: 多算法比较
        printf("9) Extra summary (toxic ratio)\n");      // Distinction: 额外统计
        printf("10) List ALL words alphabetically\n");    // Pass: 按字母排序展示（可选）
        printf("0) Back\n");
        printf("Select: ");

        if (scanf("%d", &sub) != 1) {
            // 简单清空输入
            int c; while ((c = getchar()) != '\n' && c != EOF);
            sub = -1;
            continue;
        }

        switch (sub) {
        case 1: {
            int s;
            printf("Choose source: 1=File1  2=File2  0=Auto(File1->File2): ");
            if (scanf("%d", &s) == 1 && (s == 0 || s == 1 || s == 2)) {
                g_use_file = s;
            }
            else {
                printf("Invalid source. Keep current.\n");
            }
        } break;
        case 2: {
            //设置排序键（频率/字母）
            int k;
            printf("Choose key: 1=Frequency desc (ties->alpha), 2=Alphabetical : ");
            if (scanf("%d", &k) == 1) g_key = (k == 2 ? KEY_ALPHA : KEY_FREQ_DESC);
        } break;
        case 3: {
            //设置算法（冒泡/快速/归并）
            int a;
            printf("Choose algorithm: 1=Bubble  2=Quick  3=Merge : ");
            if (scanf("%d", &a) == 1) {
                if (a == 1) g_alg = ALG_BUBBLE;
                else if (a == 3) g_alg = ALG_MERGE;
                else g_alg = ALG_QUICK;
            }
        } break;
        case 4: 
            g_use_secondary_tiebreak = !g_use_secondary_tiebreak;
            printf("Secondary tiebreak is now %s\n",
                g_use_secondary_tiebreak ? "ON" : "OFF");
            break;
        case 5: {
            //设置 Top N
            int n;
            printf("Enter Top N (>=1): ");
            if (scanf("%d", &n) == 1 && n > 0) g_topN = n;
        } break;
        case 6:
            //显示 Top N（全词）
            sort_and_show_topN_all(g_key, g_alg, g_topN);
            break;
        case 7:
            // 毒词通常只看频率高→低；若你要支持字母序也可传 g_key
            sort_and_show_topN_toxic(g_alg, g_topN);
            break;
        case 8:
            compare_algorithms_topN(g_topN);
            break;
        case 9:
            show_extra_summary();
            break;
        case 10:
            list_alpha_all();
            break;
        case 0:
            break;
        default:
            printf("Invalid choice.\n");
        }
    } while (sub != 0);
}

// 只比较“主键”
static int cmp_primary(const Pair* a, const Pair* b, SortKey key) {
    if (key == KEY_FREQ_DESC) {
        if (a->count != b->count) return (b->count - a->count); // 频率降序
        return 0; // 同频不决定先后
    }
    else { // KEY_ALPHA
        int s = strcmp(a->word, b->word);
        if (s != 0) return s;
        return 0; // 同字母不决定先后
    }
}

// 主比较器：可切换是否使用二级键
static int cmp_pairs(const Pair* a, const Pair* b, SortKey key) {
    int c = cmp_primary(a, b, key);
    if (c != 0 || !g_use_secondary_tiebreak) return c;

    // 二级键（你之前的行为）
    if (key == KEY_FREQ_DESC) {
        return strcmp(a->word, b->word);           // 同频时按字母序
    }
    else { // KEY_ALPHA
        return (b->count - a->count);              // 同字母时频率高的在前
    }
}

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

static int partition_pairs(Pair a[], int l, int r, SortKey key) {
    Pair pivot = a[r];               g_stats.moves++; // 读到寄存器算一次
    int i = l - 1;
    for (int j = l; j < r; ++j) {
        if (cmp_with_stats(&a[j], &pivot, key) <= 0) {
            ++i; swap_pair(&a[i], &a[j]);
        }
    }
    swap_pair(&a[i + 1], &a[r]);
    return i + 1;
}
static void quick_sort_pairs(Pair a[], int l, int r, SortKey key) {
    if (l >= r) return;
    int p = partition_pairs(a, l, r, key);
    quick_sort_pairs(a, l, p - 1, key);
    quick_sort_pairs(a, p + 1, r, key);
}

// merge 里统计写入
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

static void merge_sort_pairs(Pair a[], int l, int r, SortKey key) {
    if (l >= r) return;
    int m = (l + r) / 2;
    merge_sort_pairs(a, l, m, key);
    merge_sort_pairs(a, m + 1, r, key);
    merge_pairs(a, l, m, r, key);
}


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

static void pick_tokens(char (**out_words)[50], int* out_count) {
    if (g_use_file == 1 && file1Loaded) { *out_words = words1; *out_count = wordCount1; return; }
    if (g_use_file == 2 && file2Loaded) { *out_words = words2; *out_count = wordCount2; return; }
    // Auto 模式：沿用你现在的优先级
    if (file1Loaded) { *out_words = words1; *out_count = wordCount1; return; }
    if (file2Loaded) { *out_words = words2; *out_count = wordCount2; return; }
    *out_words = NULL; *out_count = 0;
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
    // 去引号再打开文件
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
        printf("[!] Error: Cannot create output file '%s'\n", path);
        printf("Recovery Guide:\n");
        printf("1. Make sure the file name is valid\n");
        printf("2. Check if you have write permissions in the target directory\n");
        printf("3. Ensure the directory exists\n");
        printf("4. Try a simpler file name in the current directory\n");
        printf("5. Examples: \"results.txt\", \"analysis_report.txt\"\n");
        printf("6. Close the file if it's already open in another program\n");
        perror("Detailed error");
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

void displayNegativityScale() {
    printf("Negativity scale analysis (placeholder)\n");
}

void displayWordOccurrence() {
    printf("Word occurrence analysis (placeholder)\n");
}

void displayFileRecoveryTips(const char* problemType) {
    printf("\nFile Recovery Instructions\n");

    if (strcmp(problemType, "file_not_found") == 0) {
        printf("Problem: File not found or cannot be opened\n");
        printf("Solutions:\n");
        printf("1. Check spelling of file name\n");
        printf("2. Move file to program's directory\n");
        printf("3. Use full path: C:/folder/file.txt\n");
        printf("4. For spaces, use quotes: \"my file.txt\"\n");
    }
    else if (strcmp(problemType, "permission") == 0) {
        printf("Problem: File permission denied\n");
        printf("Solutions:\n");
        printf("1. Run program as administrator\n");
        printf("2. Check file/folder permissions\n");
        printf("3. Try a different directory\n");
    }
    else if (strcmp(problemType, "corrupted") == 0) {
        printf("Problem: File appears corrupted\n");
        printf("Solutions:\n");
        printf("1. Verify file is not empty\n");
        printf("2. Ensure it's a valid text/CSV file\n");
        printf("3. Try opening in another program\n");
        printf("4. Check file encoding (should be UTF-8 or ANSI)\n");
    }

    printf("Quick fix: Place file in directory as this program and use just the file name\n\n");
}
