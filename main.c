#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <ctype.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>

#define MAX_WORDS 40000
#define MAX_WORD_LENGTH 50
#define MAX_STOPWORDS 500
#define MAX_TEXT_LENGTH 200000
#define MAX_VARIANTS 300
#define MAX_TOXIC_WORDS 1000
#define MAX_PHRASES 500
#define MAX_TEXT_LENGTH_ADV 50000

// ====== STAGE 2 数据结构 ======
struct WordInfo {
    char word[MAX_WORD_LENGTH];
    int count;
};

struct VariantMap {
    char variant[MAX_WORD_LENGTH];
    char standard[MAX_WORD_LENGTH];
};
// ========== STAGE 2 数据结构结束 ==========

// ========== STAGE 3 数据结构 ==========
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
// ========== STAGE 3 数据结构结束 ==========

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

    // ========== STAGE 3 字段 ==========
    struct ToxicWord toxic_words_list[MAX_TOXIC_WORDS];
    struct ToxicPhrase toxic_phrases_list[MAX_PHRASES];
    int toxic_words_count;
    int toxic_phrases_count;
    int total_toxic_occurrences;
    int severity_count[6]; // 1-5 for severity levels
    float toxicity_density;
    int bigram_toxic_occurrences;
    int trigram_toxic_occurrences;
    // ========== STAGE 3 字段结束 ==========
};

// ====== 全局变量 ======
struct AnalysisData analysis_data;
char current_filename[256] = "";
char current_manual_filtered_filename[256] = "";
bool user_manually_saved_current_session = false;
static int g_toxic_loaded = 0;
static int g_toxic_count = 0;
static char g_toxic[500][MAX_WORD_LENGTH];

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
#define ISALPHA(c) isalpha((unsigned char)(c))
#define TOLOWER(c) tolower((unsigned char)(c))

//高级分析时使用，记录唯一词及其频次
typedef enum { KEY_FREQ_DESC, KEY_ALPHA } SortKey;  //排序"键"（频率降序 / 字母序）
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
static SortAlg g_alg = ALG_BUBBLE;       // 默认bubble
static int     g_topN = 10;             // 默认Top N
static int g_use_secondary_tiebreak = 1; // 1=使用二级键(现在的行为), 0=纯主键，制造ties观察稳定性
static int g_use_file = 1; // 1=File1, 2=File2, 0=auto(现状)

// ====== 函数声明 ======
char get_menu_option(const char* valid_options, const char* prompt);

// ========== STAGE 2 函数声明 ==========
int read_line(char* buf, size_t cap);
int load_stopwords(char stopwords[][MAX_WORD_LENGTH]);
int is_stopword(char* word, char stopwords[][MAX_WORD_LENGTH], int stop_count);
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
// ========== STAGE 2 函数声明结束 ==========

// ========== STAGE 3 函数声明 ==========
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

// Stage 3 菜单函数
void display_toxic_menu();
void toxic_analysis();
void dictionary_management();
void save_toxic_dictionary(const char* filename);
void view_all_toxic_words();

// Stage 3 工具函数
void calculate_toxicity_density();
void add_custom_toxic_word();
void remove_toxic_word();
int phrase_contains_toxic_words(const char* phrase, char found_words[][MAX_WORD_LENGTH], int max_found, int* max_severity);
void add_custom_toxic_phrase(const char* phrase);
// ========== STAGE 3 函数声明结束 ==========

// -------- 函数声明（更新） --------
void loadTextFile(int fileNumber);
void displayToxicWordAnalysis();
void sortAndDisplayTopNWords();
void saveResultsToFile();
void loadDictionaries();
void reload_dictionaries();
void handleError(const char* message);
void displayNegativityScale();
void displayWordOccurrence();
void showFileHistory(int fileNumber);
void handleFileMenu();
bool isCSVFile(const char* filename);
void processCSVFile(FILE* f, char (*targetWords)[50], int* targetWordCount);
bool isFileCorrupted(const char* filePath, const char* fileContent, size_t contentSize);
void displayFileRecoveryTips(const char* problemType);
static int  build_pairs_from_tokens(char (*words)[50], int wordCount, Pair out[], int maxOut);
static void sort_pairs(Pair a[], int n, SortKey key, SortAlg alg);
static int  load_toxicwords(void);
void        menu_sort_and_report(void);
static void merge_sort_pairs(Pair a[], int l, int r, SortKey key);

// ====== 新增：高级文本分析函数声明 ======
int tokenize_and_clean(char* text, struct WordInfo words[], char stopwords[][MAX_WORD_LENGTH], int stop_count, int* total_chars);
int count_sentences(char* text);
void generate_statistics(struct WordInfo words[], int word_count, int total_words, int total_chars, int sentences, int stopwords_removed);
void show_top_words(struct WordInfo words[], int word_count, int n);
void advancedTextAnalysis();
void sort_and_show_topN_all(SortKey key, SortAlg alg, int topN);
void sort_and_show_topN_toxic(SortAlg alg, int topN);
void compare_algorithms_topN(int topN);
void show_extra_summary(void);
void list_alpha_all(void);

int main() {
    int userChoice;

    for (;;) {
        printf("\n\nToxic Word Text Analyser Menu\n");
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
            } else {
                display_advanced_analysis_menu();
            }
            break;
        case 3:
            if (!file1Loaded && !file2Loaded) {
                printf("No files loaded. Please use option 1 first to load files.\n");
            } else {
                display_toxic_menu();
            }
            break;
        case 4:
            if (!file1Loaded && !file2Loaded) {
                printf("No files loaded. Please use option 1 first to load files.\n");
            } else {
                menu_sort_and_report();
            }
            break;
        case 5:
            printf("Enter output file path (e.g., analysisReport.csv):\n>");
            if (!read_line(outputFilePath, sizeof(outputFilePath))) {
                handleError("Failed to read output path");
                break;
            }
            saveResultsToFile();
            break;
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

// ========== STAGE 2 函数实现 ==========

char get_menu_option(const char* valid_options, const char* prompt) {
    char buffer[100];
    char option;

    while (1) {
        printf("%s", prompt);

        if (!read_line(buffer, sizeof(buffer))) {
            printf("Input error. Please try again.\n");
            continue;
        }

        // 检查是否为空输入
        if (strlen(buffer) == 0) {
            printf("Please enter a option.\n");
            continue;
        }

        // 检查是否只输入了一个字符
        if (strlen(buffer) != 1) {
            printf("Error: Please enter exactly one character.\n");
            continue;
        }

        // 只取第一个字符
        option = buffer[0];

        // 检查是否是有效选择
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

// ====== 核心功能函数 ======

// 安全读取一行输入
int read_line(char* buf, size_t cap) {
    if (!fgets(buf, (int)cap, stdin)) return 0;
    size_t n = strlen(buf);
    while (n && (buf[n - 1] == '\n' || buf[n - 1] == '\r'))
        buf[--n] = '\0';
    return 1;
}

// 加载停用词列表
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

// 检查是否为停用词
int is_stopword(char* word, char stopwords[][MAX_WORD_LENGTH], int stop_count) {
    for (int i = 0; i < stop_count; i++) {
        if (strcmp(word, stopwords[i]) == 0) {
            return 1;
        }
    }
    return 0;
}

// 变体映射功能
void init_basic_variants() {
    // 只保留绝对必要的核心映射，其他都放在外部文件中
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
    // 从文件加载额外的映射
    int loaded = load_variant_mappings("variant_mappings.txt");
    printf("Initialised %d core variants + loaded %d mappings from file (total %d, limit %d)\n",
        num_core, loaded, analysis_data.variant_count, MAX_VARIANTS);
}

int load_variant_mappings(const char* filename) {
    FILE* file = fopen(filename, "r");
    if (file == NULL) {
        return 0;
    }

    int loaded = 0;
    char line[256];
    int is_first_line = 1;  // 跳过表头

    while (fgets(line, sizeof(line), file) && analysis_data.variant_count < MAX_VARIANTS) {
        if (line[0] == '\n' || line[0] == '#' || line[0] == '\r') {
            continue;
        }

        line[strcspn(line, "\n")] = '\0';
        line[strcspn(line, "\r")] = '\0';

        // 跳过表头行
        if (is_first_line && (strstr(line, "Input Form") != NULL || strstr(line, "Normalised Form") != NULL)) {
            is_first_line = 0;
            continue;
        }
        is_first_line = 0;

        char* separator = strchr(line, '=');
        if (separator == NULL) {
            continue;
        }

        *separator = '\0';
        char* variant = line;
        char* standard = separator + 1;

        while (*variant == ' ') variant++;
        while (*standard == ' ') standard++;

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
void toggle_variant_processing_with_examples() {
    // First show current status
    printf("\nCurrent Text Normalisation: %s\n", 
           analysis_data.variant_processing_enabled ? "ENABLED" : "DISABLED");
    
    // Show hardcoded examples in a table
    printf("\n=== TEXT NORMALISATION EXAMPLES ===\n");
    
    // Hardcoded examples - always show these
    const char* examples[][2] = {
        {"u", "you"},
        {"ur", "your"}, 
        {"r", "are"},
        {"btw", "by the way"},
        {"omg", "oh my god"},
        {"lol", "laughing out loud"},
        {"thx", "thanks"},
        {"plz", "please"},
        {"gr8", "great"},
        {"2day", "today"}
    };
    
    int example_count = sizeof(examples) / sizeof(examples[0]);
    
    // Display the examples in a nice table
    printf("Examples of text normalization:\n");
    printf("+-----------------+-----------------------+\n");
    printf("| Short Form      | Expanded To           |\n");
    printf("+-----------------+-----------------------+\n");
    
    for (int i = 0; i < example_count; i++) {
        printf("| %-15s | %-21s |\n", 
               examples[i][0], 
               examples[i][1]);
    }
    printf("+-----------------+-----------------------+\n");
    
    // Show a few of the actual mappings from the loaded file
    if (analysis_data.variant_count > 0) {
        printf("\nSample of available mappings:\n");
        printf("+-----------------+-----------------------+\n");
        printf("| Input Form      | Normalised Form       |\n");
        printf("+-----------------+-----------------------+\n");
        
        int mappings_shown = 0;
        for (int i = 0; i < analysis_data.variant_count && mappings_shown < 8; i++) {
            printf("| %-15s | %-21s |\n",
                   analysis_data.variant_mappings[i].variant,
                   analysis_data.variant_mappings[i].standard);
            mappings_shown++;
        }
        printf("+-----------------+-----------------------+\n");
        
        if (analysis_data.variant_count > 8) {
            printf("... and %d more mappings available\n", analysis_data.variant_count - 8);
        }
    }
    
    // Ask user if they want to toggle normalization
    printf("\nDo you want to %s text normalization? (y/n): ",
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
                printf("  * Change due to normalization: %+d words\n", word_change);
            }
        }
    } else {
        printf("Text Normalisation setting unchanged.\n");
    }
}

void add_token_to_analysis(const char* tok, int* removed_by_stopwords) {
    if (!tok || !*tok) return;

    // 检查是否包含字母
    int has_letters = 0;
    for (int j = 0; tok[j]; j++) {
        if (ISALPHA(tok[j])) { has_letters = 1; break; }
    }
    if (!has_letters) return;

    // 检查停用词
    if (is_stopword((char*)tok, analysis_data.stopwords, analysis_data.stop_count)) {
        (*removed_by_stopwords)++;
        return;
    }

    // 添加到过滤词列表
    if (analysis_data.filtered_word_list != NULL &&
        analysis_data.filtered_word_count < MAX_WORDS) {
        strncpy(analysis_data.filtered_word_list[analysis_data.filtered_word_count],
            tok, MAX_WORD_LENGTH - 1);
        analysis_data.filtered_word_list[analysis_data.filtered_word_count][MAX_WORD_LENGTH - 1] = '\0';
    }
    analysis_data.filtered_word_count++;
    analysis_data.total_words_filtered++;
    analysis_data.total_chars += (int)strlen(tok);

    // 更新词频统计
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

// 动态重新处理文本
void reprocess_with_variants() {
    if (analysis_data.original_word_list == NULL) return;

    // 重置统计
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

        // 应用变体映射（若启用）
        char* normalised = normalise_variant(current_word);

        if (normalised != current_word) {
            variants_normalised++;

            // 短语映射：拆分为多个词逐个处理
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

            // 单词映射
            strncpy(current_word, normalised, MAX_WORD_LENGTH - 1);
            current_word[MAX_WORD_LENGTH - 1] = '\0';
        }

        // 检查是否包含字母
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

    // 只在需要时打印变体扩展信息（现在由调用者控制）
    if (analysis_data.variant_processing_enabled && variants_normalised > 0) {
        printf("  - Text forms normalised: %d (abbreviations and Leet Speak)\n", variants_normalised);
    }
}

// 处理文本文件
void process_text_file(const char* filename) {
    printf("\nProcessing file: %s\n", filename);
    strncpy(current_filename, filename, sizeof(current_filename) - 1);
    current_filename[sizeof(current_filename) - 1] = '\0';

    // 清理旧数据
    cleanup_analysis_data();

    // 重置手动保存标志
    user_manually_saved_current_session = false;

    // 加载停用词
    analysis_data.stop_count = load_stopwords(analysis_data.stopwords);
    if (analysis_data.stop_count == 0) {
        printf("Cannot continue without stopwords.\n");
        return;
    }

    // 打开文件
    FILE* file = fopen(filename, "r");
    if (file == NULL) {
        printf("ERROR: Cannot open file: %s\n", filename);
        return;
    }

    // 分配文本缓冲区
    analysis_data.text = (char*)malloc(MAX_TEXT_LENGTH);
    if (!analysis_data.text) {
        printf("Error: Memory allocation failed\n");
        fclose(file);
        return;
    }
    memset(analysis_data.text, 0, MAX_TEXT_LENGTH);

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

        // 统计原始词数
        char line_copy[1000];
        strcpy(line_copy, line);
        char* token_tmp = strtok(line_copy, DELIMS);
        while (token_tmp) {
            analysis_data.total_words_original++;
            token_tmp = strtok(NULL, DELIMS);
        }
    }
    fclose(file);

    // 处理非ASCII字符
    for (size_t i = 0; analysis_data.text[i]; ++i) {
        unsigned char ch = (unsigned char)analysis_data.text[i];
        if (ch > 127) analysis_data.text[i] = ' ';
    }

    printf("File reading completed. Total words in file: %d\n", analysis_data.total_words_original);
    if (analysis_data.text[0] == '\0') {
        printf("ERROR: No content read from file\n");
        return;
    }

    // 分配内存
    analysis_data.words = (struct WordInfo*)calloc(MAX_WORDS, sizeof(struct WordInfo));
    if (!analysis_data.words) {
        printf("Error: Memory allocation failed\n");
        return;
    }

    analysis_data.filtered_word_list = (char**)calloc(MAX_WORDS, sizeof(char*));
    if (!analysis_data.filtered_word_list) {
        printf("Error: Memory allocation failed\n");
        free(analysis_data.words);
        analysis_data.words = NULL;
        return;
    }

    for (int i = 0; i < MAX_WORDS; i++) {
        analysis_data.filtered_word_list[i] = (char*)malloc(MAX_WORD_LENGTH);
        if (!analysis_data.filtered_word_list[i]) {
            for (int j = 0; j < i; j++) free(analysis_data.filtered_word_list[j]);
            free(analysis_data.filtered_word_list);
            analysis_data.filtered_word_list = NULL;
            printf("Error: Memory allocation failed\n");
            free(analysis_data.words);
            analysis_data.words = NULL;
            return;
        }
    }

    // 分配原始词列表
    analysis_data.original_word_list = (char**)calloc(MAX_WORDS, sizeof(char*));
    if (!analysis_data.original_word_list) {
        printf("Error: Memory allocation failed for original word list\n");
        return;
    }

    for (int i = 0; i < MAX_WORDS; i++) {
        analysis_data.original_word_list[i] = (char*)malloc(MAX_WORD_LENGTH);
        if (!analysis_data.original_word_list[i]) {
            for (int j = 0; j < i; j++) free(analysis_data.original_word_list[j]);
            free(analysis_data.original_word_list);
            analysis_data.original_word_list = NULL;
            printf("Error: Memory allocation failed for original word list\n");
            return;
        }
    }

    // 分词处理
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

    // 收集所有原始词
    while (token && analysis_data.original_word_count < MAX_WORDS) {
        char clean_word[MAX_WORD_LENGTH];
        strncpy(clean_word, token, MAX_WORD_LENGTH - 1);
        clean_word[MAX_WORD_LENGTH - 1] = '\0';

        // 去掉特殊前缀
        if (clean_word[0] == '#' || clean_word[0] == '@') {
            memmove(clean_word, clean_word + 1, strlen(clean_word));
        }

        // 小写化并过滤非ASCII
        for (int i = 0; clean_word[i]; i++) {
            unsigned char ch = (unsigned char)clean_word[i];
            if (ch > 127) {
                clean_word[i] = '\0';
                break;
            }
            clean_word[i] = (char)tolower(ch);
        }

        // 对 original_word_list 应用变体处理
        char* filtered_word = clean_word;
        if (analysis_data.variant_processing_enabled) {
            filtered_word = normalise_variant(clean_word);

            // 如果变体映射为短语，拆分为多个词
            if (strchr(filtered_word, ' ') != NULL && filtered_word != clean_word) {
                char phrase_buf[MAX_WORD_LENGTH * 4];
                strncpy(phrase_buf, filtered_word, sizeof(phrase_buf) - 1);
                phrase_buf[sizeof(phrase_buf) - 1] = '\0';

                char* part = strtok(phrase_buf, " ");
                while (part && analysis_data.original_word_count < MAX_WORDS) {
                    if (strlen(part) > 0) {
                        strncpy(analysis_data.original_word_list[analysis_data.original_word_count],
                            part, MAX_WORD_LENGTH - 1);
                        analysis_data.original_word_list[analysis_data.original_word_count][MAX_WORD_LENGTH - 1] = '\0';
                        analysis_data.original_word_count++;
                    }
                    part = strtok(NULL, " ");
                }
                continue; // 跳过下面的单个词添加
            }
        }

        // 保存处理后的原始词（包含变体处理，但不排除停用词）
        if (strlen(filtered_word) > 0) {
            strncpy(analysis_data.original_word_list[analysis_data.original_word_count],
                filtered_word, MAX_WORD_LENGTH - 1);
            analysis_data.original_word_list[analysis_data.original_word_count][MAX_WORD_LENGTH - 1] = '\0';
            analysis_data.original_word_count++;
        }

        token = strtok(NULL, DELIMS);
    }

    free(text_copy);

    // 应用变体处理和过滤
    reprocess_with_variants();

    // 句子统计
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
}

// ====== 分析功能函数 ======

void word_analysis() {
    if (analysis_data.text == NULL) {
        printf("No file filtered. Use option 1 first.\n");
        return;
    }

    printf("\n=== GENERAL WORD STATISTICS WITH ADVANCED ANALYSIS ===\n");
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

    // 高频词显示
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

// 自动保存过滤词列表（不显示提示）
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
        // 静默保存，不显示消息
    }
}

void save_filtered_word_list() {
    if (analysis_data.text == NULL) {
        printf("No file filtered. Use option 1 first.\n");
        return;
    }

    char filename[256];
    printf("Enter filename (or press Enter for 'filteredWords.txt'): ");
    if (!read_line(filename, sizeof(filename))) {
        printf("Failed to read filename.\n");
        return;
    }

    if (strlen(filename) == 0) {
        strcpy(filename, "filtered_words.txt");
    }

    if (strlen(filename) < 4 || strcmp(filename + strlen(filename) - 4, ".txt") != 0) {
        strcat(filename, ".txt");
    }

    FILE* file = fopen(filename, "w");
    if (file) {
        // 记录保存时的Text Normalisation状态
        if (analysis_data.variant_processing_enabled) {
            fprintf(file, "# TextNormalisation: enabled\n");
        }
        else {
            fprintf(file, "# TextNormalisation: disabled\n");
        }
        for (int i = 0; i < analysis_data.filtered_word_count; i++) {
            fprintf(file, "%s\n", analysis_data.filtered_word_list[i]);
        }
        fclose(file);

        strncpy(current_manual_filtered_filename, filename, sizeof(current_manual_filtered_filename) - 1);
        current_manual_filtered_filename[sizeof(current_manual_filtered_filename) - 1] = '\0';

        // 设置手动保存标志
        user_manually_saved_current_session = true;
        printf("Filtered words saved to: %s (%d words)\n", filename, analysis_data.filtered_word_count);
    }
    else {
        printf("Could not save to: %s\n", filename);
    }
}

// ====== 工具函数 ======

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

    //重置手动保存标志
    user_manually_saved_current_session = false;

    // 重置所有计数器
    analysis_data.word_count = 0;
    analysis_data.total_words_filtered = 0;
    analysis_data.total_chars = 0;
    analysis_data.sentences = 0;
    analysis_data.stopwords_removed = 0;
    analysis_data.total_words_original = 0;
    analysis_data.filtered_word_count = 0;
    analysis_data.original_word_count = 0;
    analysis_data.text_filtered = false;

    // 重置 Stage 3 相关计数器
    analysis_data.total_toxic_occurrences = 0;
    analysis_data.toxicity_density = 0.0;
    analysis_data.bigram_toxic_occurrences = 0;
    analysis_data.trigram_toxic_occurrences = 0;
    memset(analysis_data.severity_count, 0, sizeof(analysis_data.severity_count));
}

void display_advanced_analysis_menu() {
    char option;
    do {
        printf("\n=== ADVANCED TEXT ANALYSIS MENU ===\n");
        printf("1. Word Analysis with Stopwords Filtering\n");
        printf("2. Text Normalisation (Toggle & View Examples)\n");
        printf("3. Save Filtered Word List\n");
        printf("4. Back to Main Menu\n");
        option = get_menu_option("1234", "Enter your option (1-4): ");

        switch (option) {
        case '1':
            // Check if files are loaded using the dual-file system
            if (!file1Loaded && !file2Loaded) {
                printf("No files loaded. Please use option 1 first to load files.\n");
            } else {
                // Process the first loaded file for advanced analysis
                if (file1Loaded) {
                    printf("Processing File 1: %s\n", inputFilePath1);
                    process_text_file(inputFilePath1);
                    word_analysis();
                } else if (file2Loaded) {
                    printf("Processing File 2: %s\n", inputFilePath2);
                    process_text_file(inputFilePath2);
                    word_analysis();
                }
            }
            break;
        case '2':
            if (!analysis_data.text_filtered) {
                printf("No file processed for analysis. Please use option 1 first.\n");
            } else {
                // Combined functionality: toggle + show examples
                toggle_variant_processing_with_examples();
            }
            break;
        case '3':
            if (!analysis_data.text_filtered) {
                printf("No file processed for analysis. Please use option 1 first.\n");
            } else {
                save_filtered_word_list();
            }
            break;
        case '4':
            printf("Returning to main menu...\n");
            break;
        default:
            printf("Invalid option. Please enter 1-4.\n");
        }
    } while (option != '4');
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

// 加载毒性数据
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
        // 去掉换行
        line[strcspn(line, "\n")] = '\0';
        line[strcspn(line, "\r")] = '\0';

        if (strlen(line) == 0 || line[0] == '#') continue;

        // 解析: word,severity
        char* token = strtok(line, ",");
        if (!token) continue;

        char word[100];
        strcpy(word, token);

        token = strtok(NULL, ",");
        if (!token) {
            printf("Warning: Invalid line (no severity): %s\n", line);
            continue;
        }
        int severity = atoi(token);
        if (severity < 1 || severity > 5) severity = 3;

        // 判断是否为 phrase
        if (strchr(word, ' ') != NULL) {
            if (analysis_data.toxic_phrases_count < MAX_PHRASES) {
                int idx = analysis_data.toxic_phrases_count;

                // 统计单词数
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
            // 单词
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

// ====== MERIT: Dynamic Dictionary Reloading ======
void reload_dictionaries() {
    printf("\n=== RELOAD DICTIONARIES ===\n");
    
    // Reload stopwords
    int old_stop_count = analysis_data.stop_count;
    analysis_data.stop_count = load_stopwords(analysis_data.stopwords);
    printf("Stopwords: %d -> %d words\n", old_stop_count, analysis_data.stop_count);
    
    // Reload toxic words
    int old_toxic_count = analysis_data.toxic_words_count;
    int old_phrase_count = analysis_data.toxic_phrases_count;
    
    analysis_data.toxic_words_count = 0;
    analysis_data.toxic_phrases_count = 0;
    load_toxic_data("toxicwords.txt");
    
    printf("Toxic words: %d -> %d words\n", old_toxic_count, analysis_data.toxic_words_count);
    printf("Toxic phrases: %d -> %d phrases\n", old_phrase_count, analysis_data.toxic_phrases_count);
    
    // Sync with Stage 4 system
    sync_toxic_systems();
    
    printf("All dictionaries reloaded successfully!\n");
    
    // Reprocess text if we have loaded files
    if (analysis_data.text_filtered) {
        printf("Re-processing text with updated dictionaries...\n");
        reprocess_with_variants();
    }
}

// 不区分大小写的字符串比较
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

// 检查文件是否存在
bool file_exists(const char* filename) {
    FILE* file = fopen(filename, "r");
    if (file) {
        fclose(file);
        return true;
    }
    return false;
}

// 检查是否为有毒词汇
// Check if a word is toxic (works for both Stage 3 and Stage 4)
int is_toxic_word(const char* word) {
    // Check Stage 3 system first
    for (int i = 0; i < analysis_data.toxic_words_count; i++) {
        if (string_case_insensitive_compare(word, analysis_data.toxic_words_list[i].word) == 0) {
            return 1;
        }
    }
    
    // Also check Stage 4 system as backup
    for (int i = 0; i < g_toxic_count; i++) {
        if (string_case_insensitive_compare(word, g_toxic[i]) == 0) {
            return 1;
        }
    }
    
    return 0;
}

// 获取毒性严重程度
int get_toxic_severity(const char* word) {
    for (int i = 0; i < analysis_data.toxic_words_count; i++) {
        if (string_case_insensitive_compare(word, analysis_data.toxic_words_list[i].word) == 0) {
            return analysis_data.toxic_words_list[i].severity;
        }
    }
    return 0;
}

// 检测毒性内容
void detect_toxic_content(const char* word) {
    if (is_toxic_word(word)) {
        analysis_data.total_toxic_occurrences++;
        int severity = get_toxic_severity(word);

        // 更新毒性词汇频率
        for (int i = 0; i < analysis_data.toxic_words_count; i++) {
            if (string_case_insensitive_compare(word, analysis_data.toxic_words_list[i].word) == 0) {
                analysis_data.toxic_words_list[i].frequency++;
                break;
            }
        }

        // 更新严重程度统计
        if (severity >= 1 && severity <= 5) {
            analysis_data.severity_count[severity]++;
        }
    }
}

// 检测毒性短语
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

                char found_words[10][MAX_WORD_LENGTH];
                int max_sev_in_phrase = 0;
                int toxic_word_count = phrase_contains_toxic_words(
                    analysis_data.toxic_phrases_list[j].phrase,
                    found_words,
                    10,
                    &max_sev_in_phrase
                );

                int sev = analysis_data.toxic_phrases_list[j].severity;
                if (sev < 1 || sev > 5) sev = 3;

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

                    char found_words[10][MAX_WORD_LENGTH];
                    int max_sev_in_phrase = 0;
                    int toxic_word_count = phrase_contains_toxic_words(
                        analysis_data.toxic_phrases_list[j].phrase,
                        found_words,
                        10,
                        &max_sev_in_phrase
                    );

                    int sev = analysis_data.toxic_phrases_list[j].severity;
                    if (sev < 1 || sev > 5) sev = 3;

                    analysis_data.toxic_phrases_list[j].frequency++;
                    analysis_data.trigram_toxic_occurrences++;

                    break;
                }
            }
        }
    }
}

// 检查短语是否包含毒性词汇
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

// 重置毒性计数
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

// 计算毒性密度
void calculate_toxicity_density() {
    if (analysis_data.total_words_filtered > 0) {
        analysis_data.toxicity_density = (float)analysis_data.total_toxic_occurrences /
            analysis_data.total_words_filtered * 100;
    }
    else {
        analysis_data.toxicity_density = 0.0;
    }
}

// 运行毒性分析
void run_toxic_analysis() {
    if (!analysis_data.text_filtered) {
        printf("No text filtered. Please load and process a file first.\n");
        return;
    }

    char filename_to_use[256];
    bool need_reprocess = false;

    if (user_manually_saved_current_session) {
        const char* manually_saved_filename =
            strlen(current_manual_filtered_filename) > 0 ?
            current_manual_filtered_filename :
            "filtered_words.txt";

        FILE* file = fopen(manually_saved_filename, "r");
        bool saved_without_normalisation = false;
        bool found_metadata = false;

        if (file) {
            char line[MAX_WORD_LENGTH];
            // 检查第一行是否有元数据
            if (fgets(line, sizeof(line), file)) {
                line[strcspn(line, "\n")] = 0;
                if (strstr(line, "# TextNormalisation: disabled") != NULL) {
                    saved_without_normalisation = true;
                    found_metadata = true;
                }
                else if (strstr(line, "# TextNormalisation: enabled") != NULL) {
                    saved_without_normalisation = false;
                    found_metadata = true;
                }
            }

            // 如果没有元数据，使用备用检测方法
            if (!found_metadata) {
                rewind(file);
                int checked_lines = 0;
                while (fgets(line, sizeof(line), file) && checked_lines < 10) {
                    line[strcspn(line, "\n")] = 0;
                    // 跳过注释行
                    if (line[0] == '#') continue;

                    // 检查是否包含常见的未归一化形式
                    if (strcmp(line, "omg") == 0 || strcmp(line, "lol") == 0 ||
                        strcmp(line, "btw") == 0 || strcmp(line, "ur") == 0 ||
                        strcmp(line, "thx") == 0 || strcmp(line, "plz") == 0 ||
                        strcmp(line, "u") == 0 || strcmp(line, "r") == 0) {
                        saved_without_normalisation = true;
                        break;
                    }
                    checked_lines++;
                }
            }
            fclose(file);
        }

        if (saved_without_normalisation) {
            printf("\nWARNING: You saved a filtered word list WITHOUT Text Normalisation.\n");
            printf("This means abbreviations and Leet Speak won't be expanded.\n");
            printf("Examples in your file: 'omg', 'lol', 'btw' will remain as-is.\n");
            printf("\nDo you want to:\n");
            printf("1. Use the non-normalised file (faster, but less accurate)\n");
            printf("2. Re-process with Text Normalisation (recommended for better accuracy)\n");
            printf("Enter your option (1-2): ");

            char option;
            scanf("%c", &option);
            getchar();

            if (option == '1') {
                printf("\nUsing non-normalised file for analysis: %s\n", manually_saved_filename);
                strcpy(filename_to_use, manually_saved_filename);
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
            printf("Using your manually saved file: %s\n", manually_saved_filename);
            strcpy(filename_to_use, manually_saved_filename);
        }
    }
    else {
        // 用户没有手动保存 - 自动处理
        printf("No manually saved filtered word list detected.\n");
        printf("Auto-saving normalised word list for toxic analysis...\n");

        // 确保使用最佳配置
        if (!analysis_data.variant_processing_enabled) {
            printf("Enabling Text Normalisation for better accuracy...\n");
            analysis_data.variant_processing_enabled = true;
            need_reprocess = true;
        }

        strcpy(filename_to_use, "filtered_words_auto_saved.txt");
    }

    // 如果需要重新处理
    if (need_reprocess) {
        reprocess_with_variants();
    }

    // 自动保存当前状态的文件（如果使用的不是手动保存的文件）
    if (!user_manually_saved_current_session || strcmp(filename_to_use, "filtered_words_normalised.txt") == 0) {
        save_filtered_word_list_auto(filename_to_use);
        printf("Auto-saved word list: %s\n", filename_to_use);
    }

    // 重置毒性计数并进行分析
    reset_toxic_counts();
    printf("Starting toxic analysis using: %s\n", filename_to_use);

    // 从文件读取过滤词列表进行毒性分析
    FILE* file = fopen(filename_to_use, "r");
    if (!file) {
        printf("Error: Cannot open filtered word list file: %s\n", filename_to_use);
        return;
    }

    char word[MAX_WORD_LENGTH];
    int word_count = 0;

    while (fgets(word, sizeof(word), file) && word_count < MAX_WORDS) {
        word[strcspn(word, "\n")] = 0;
        word[strcspn(word, "\r")] = 0;

        if (strlen(word) > 0) {
            detect_toxic_content(word);
            word_count++;
        }
    }
    fclose(file);

    printf("Analysed %d words from file\n", word_count);

    // 检测毒性短语（使用内存中的 original_word_list）
    detect_toxic_phrases();

    // 计算毒性密度
    calculate_toxicity_density();

    printf("Toxic analysis completed.\n");
}

// ========== STAGE 3 菜单功能实现 ==========

// 基础毒性分析
void toxic_analysis() {
    printf("\n=== TOXIC CONTENT ANALYSIS ===\n");
    printf("Detecting for toxic content...\n");
    run_toxic_analysis();

    if (analysis_data.total_toxic_occurrences==0) {
        printf("Your file contains no toxic content.\n");
        return;
    }

    printf("\n--- TOXIC CONTENT SUMMARY ---\n");
    printf("Total toxic words detected: %d\n", analysis_data.total_toxic_occurrences);
    printf("Toxicity score: %.2f%% (%d toxic words out of %d total words detected)\n",
        analysis_data.toxicity_density,
        analysis_data.total_toxic_occurrences,
        analysis_data.total_words_filtered);

    // 全部 score 都按「word-level」算
    printf(" - Word-level (single words) : %d detections\n",
        analysis_data.total_toxic_occurrences);

    // 短语只是额外信息，清楚说明不在总分里
    printf(" - Toxic bigram matches      : %d detections (not counted in total)\n",
        analysis_data.bigram_toxic_occurrences);
    printf(" - Toxic trigram matches     : %d detections (not counted in total)\n",
        analysis_data.trigram_toxic_occurrences);

    // 显示严重程度分布 - 添加简单条形图
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

    // 显示毒性词汇表格 - 按频率排序
    printf("\n--- TOXIC WORDS DETECTED ---\n");

    // 创建临时数组用于排序
    struct ToxicWord sorted_words[MAX_TOXIC_WORDS];
    int valid_count = 0;

    for (int i = 0; i < analysis_data.toxic_words_count; i++) {
        if (analysis_data.toxic_words_list[i].frequency > 0) {
            sorted_words[valid_count] = analysis_data.toxic_words_list[i];
            valid_count++;
        }
    }

    // 按频率降序排序
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
            // 计算频率数字的位数，用于居中显示
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

            // 计算前后空格数使频率居中（总宽度11个字符）
            int total_spaces = 11 - freq_digits;
            int spaces_before = total_spaces / 2;
            int spaces_after = total_spaces - spaces_before;

            printf("| %-15s |", sorted_words[i].word);

            // 输出频率前的空格
            for (int s = 0; s < spaces_before; s++) printf(" ");
            // 输出频率数字
            printf("%d", freq);
            // 输出频率后的空格
            for (int s = 0; s < spaces_after; s++) printf(" ");

            printf("|      %d       |\n", sorted_words[i].severity);
        }
        printf("+-----------------+-----------+--------------+\n");
    }
    else {
        printf("No toxic words found.\n");
    }

    // 显示毒性短语
    printf("\n--- TOXIC PHRASES (patterns only) ---\n");
    int phrase_found = 0;
    int total_phrase_occurrences = 0;

    // 先算所有短语的总出现次数
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

                // 用工具函数检查 phrase 里面有多少 toxic words
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
                        // 多个 toxic word 的固定搭配
                        printf("  Note: Multi-toxic phrase (contains %d toxic words, max severity %d)\n",
                            toxic_word_count,
                            max_sev_in_phrase > 0 ? max_sev_in_phrase : sev);
                    }
                }
                else {
                    // 完全由 non-toxic words 组成，但组合起来有毒的 phrase
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

// 查看所有毒性词汇
void view_all_toxic_words() {
    printf("\n=== TOXIC DICTIONARY OVERVIEW ===\n");
    printf("Total words: %d, Total phrases: %d\n\n",
        analysis_data.toxic_words_count, analysis_data.toxic_phrases_count);

    // 显示单词（分严重程度显示）
    printf("TOXIC WORDS BY SEVERITY LEVEL:\n");
    printf("-------------------------------\n");

    for (int severity = 1; severity <= 5; severity++) {
        printf("\nLevel %d:\n", severity);
        int count = 0;
        for (int i = 0; i < analysis_data.toxic_words_count; i++) {
            if (analysis_data.toxic_words_list[i].severity == severity) {
                printf("%-15s", analysis_data.toxic_words_list[i].word);
                count++;
                if (count % 5 == 0) printf("\n"); // 每行显示5个单词
            }
        }
        if (count % 5 != 0) printf("\n"); // 换行
        printf("Total: %d words\n", count);
    }

    // 显示短语
    printf("\nTOXIC PHRASES:\n");
    printf("--------------\n");
    int phrases_displayed = 0;
    for (int i = 0; i < analysis_data.toxic_phrases_count; i++) {
        // 检查短语是否包含已知毒性词汇
        char found_toxic_words[10][MAX_WORD_LENGTH];
        int toxic_word_count = phrase_contains_toxic_words(
            analysis_data.toxic_phrases_list[i].phrase,
            found_toxic_words,
            10,
            NULL   // 这里不需要 max severity
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

// 词典管理
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
        printf("4. Back to menu\n");
        option = get_menu_option("1234", "Enter your option (1-4): ");

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
        case '4':
            printf("Returning to menu...\n");
            break;
        default:
            printf("Invalid option. Please enter 1-4.\n");
        }
    } while (option != '4');
}

// 添加自定义毒性词
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

    // 全部转小写
    char lower_input[MAX_WORD_LENGTH * 3];
    strcpy(lower_input, new_input);
    for (int i = 0; lower_input[i]; i++) {
        lower_input[i] = (char)tolower((unsigned char)lower_input[i]);
    }

    // 判断是单词还是 phrase
    if (strchr(lower_input, ' ') == NULL) {
        // ===== 单词逻辑 =====
        // 检查是否已存在
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

        // 按字母顺序插入
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
        // ===== phrase 逻辑 =====
        add_custom_toxic_phrase(lower_input);
    }
}

void add_custom_toxic_phrase(const char* phrase) {
    if (analysis_data.toxic_phrases_count >= MAX_PHRASES) {
        printf("Toxic phrase list is full!\n");
        return;
    }

    // 拆分 phrase 成单词
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

    // 先查一遍哪些已经是 toxic word
    int is_toxic[10] = { 0 };
    int word_sev[10] = { 0 };
    for (int i = 0; i < word_count; i++) {
        int sev = get_toxic_severity(words[i]);
        if (sev > 0) {
            is_toxic[i] = 1;
            word_sev[i] = sev;
        }
    }

    // 展示 phrase 的每个单词 & 已有信息
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

    // 只问一次：让你选哪些位置是 toxic word（避免对 stopwords 一直 y/n）
    int selected[10] = { 0 };
    char line[128];

    printf("\nWhich positions are TOXIC words? \n");
    printf("Enter numbers separated by space (e.g. 2 3), or 0 if you don't want to add new toxic words:\n> ");

    if (read_line(line, sizeof(line))) {
        char* p = strtok(line, " ");
        while (p) {
            int idx = atoi(p);
            if (idx == 0) {
                // 0 表示"没有新的 toxic word"
                break;
            }
            if (idx >= 1 && idx <= word_count) {
                selected[idx - 1] = 1;
            }
            p = strtok(NULL, " ");
        }
    }

    // 对"被选中且现在还不是 toxic"的词，再问一次 severity
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

            // 加入 toxic words 字典
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

    // 最后统计：phrase 里一共有多少 toxic word、最大 severity
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
        // 情况 A：完全"干净"的组合 → 纯 context-based phrase，问整句 severity，然后存进 phrase dictionary
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
        // 情况：短语里只有 1 个 toxic word
        // 按设定：不作为独立短语存入字典，只依靠这个单词本身进行检测
        printf("\nThis phrase contains exactly ONE toxic word.\n");
        printf("It will not be stored as a toxic phrase.\n");
        printf("Toxic detection will rely on the toxic word itself only.\n");
        return;  // 不写入 phrase 列表
    }

    else {
        // 情况 C：有 2 个或以上 toxic words → 这是 multi-toxic phrase
        phrase_severity = (final_max_sev > 0) ? final_max_sev : 3;
        printf("\nPhrase contains %d toxic word(s). Phrase severity set to Level %d (max of words).\n",
            final_toxic_count, phrase_severity);
    }

    // 只有情况 A 和 C 会走到这里：把 phrase 写入字典
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

// 移除毒性词
void remove_toxic_word() {
    // 用更大的 buffer，方便接收短语
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

    // 全部转小写，方便和字典里的内容对比
    for (int i = 0; word_to_remove[i]; i++) {
        word_to_remove[i] = (char)tolower((unsigned char)word_to_remove[i]);
    }

    bool removed = false;

    // 1.优先在「单词」列表中查找
    for (int i = 0; i < analysis_data.toxic_words_count; i++) {
        if (string_case_insensitive_compare(word_to_remove,
            analysis_data.toxic_words_list[i].word) == 0) {

            // 移除该单词：后面的元素往前挪
            for (int j = i; j < analysis_data.toxic_words_count - 1; j++) {
                analysis_data.toxic_words_list[j] = analysis_data.toxic_words_list[j + 1];
            }
            analysis_data.toxic_words_count--;
            removed = true;

            printf("Removed '%s' from toxic word list.\n", word_to_remove);
            break;
        }
    }

    // 2.如果没在 word 里找到，再在「短语」列表中查
    if (!removed) {
        for (int i = 0; i < analysis_data.toxic_phrases_count; i++) {
            if (string_case_insensitive_compare(word_to_remove,
                analysis_data.toxic_phrases_list[i].phrase) == 0) {

                // 移除短语：同样把后面的往前挪
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

    // 3.如果两个列表都没找到
    if (!removed) {
        printf("'%s' not found in toxic word or phrase dictionary.\n", word_to_remove);
        return;
    }

    // 有删到东西 → 自动保存
    save_toxic_dictionary("toxicwords.txt");
    printf("Dictionary file updated.\n");
}

// 保存毒性词典到文件
void save_toxic_dictionary(const char* filename) {
    FILE* file = fopen(filename, "w");
    if (!file) {
        printf("Error: Cannot save toxic dictionary to %s\n", filename);
        return;
    }

    // 写入文件头
    fprintf(file, "# Toxic Words Dictionary\n");
    fprintf(file, "# Format: word,severity\n");
    fprintf(file, "# Severity: 1-5 (1=mild, 5=severe)\n\n");

    // 写入所有毒性词汇（包括自定义添加的）
    for (int i = 0; i < analysis_data.toxic_words_count; i++) {
        fprintf(file, "%s,%d\n",
            analysis_data.toxic_words_list[i].word,
            analysis_data.toxic_words_list[i].severity);
    }

    // 写入毒性短语
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

// ========== STAGE 3 菜单显示函数 ==========
void display_toxic_menu() {
    char option;
    do {
        printf("\n=== TOXIC CONTENT DETECTION ===\n");
        printf("1. Dictionary Management\n");
        printf("2. Reload Dictionaries (with dynamic update)\n");
        printf("3. Toxic Analysis\n");
        printf("4. Back to Main Menu\n");
        option = get_menu_option("1234", "Enter your option (1-4): ");

        switch (option) {
        case '1':
            dictionary_management();
            break;
        case '2':
            reload_dictionaries();
            break;
        case '3':
            toxic_analysis();
            break;
        case '4':
            printf("Returning to main menu...\n");
            break;
        default:
            printf("Invalid option. Please enter 1-4.\n");
        }
    } while (option != '4');
}

// ====== 新增：通用小工具（直接可用）======
// 把一行规范化为：小写 + 非字母数字转空格
static void normalize_line(char* s) {
    for (char* p = s; *p; ++p) {
        unsigned char c = (unsigned char)*p;
        if (isalnum(c)) *p = (char)tolower(c);
        else *p = ' ';
    }
}

// 允许输入路径带引号（从资源管理器复制时常见），自动剥掉
static void strip_quotes(char* s) {
    size_t n = strlen(s);
    if (n >= 2 && s[0] == '"' && s[n - 1] == '"') {
        memmove(s, s + 1, n - 2);
        s[n - 2] = '\0';
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
        fprintf(stderr, "\n[!] Failed to open file for: %s\n", tmp);
        perror("reason");
    }
    return f;
}

//file corruption detection - FIXED VERSION
bool isFileCorrupted(const char* filePath, const char* fileContent, size_t contentSize) {
    //prevent crash for empty file
    if (contentSize==0) {
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

    FILE* f = open_file_read(filePath);
    if (!f) {
        printf("Recovery Guide:\n");
        printf("1. Make sure the file name is correct\n");
        printf("2. Move the file to the same directory as this program\n");
        return;
    }
    *targetWordCount = 0;

    // Remove quotes for file type detection
    char cleanPath[256];
    strncpy(cleanPath, filePath, sizeof(cleanPath) - 1);
    cleanPath[sizeof(cleanPath) - 1] = '\0';
    strip_quotes(cleanPath);

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
        // Original text file processing logic
        char line[4096];
        while (fgets(line, sizeof(line), f)) {
            normalize_line(line); // Lowercase + non-alphanumeric = space
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
    
    // Additional check: if no words were loaded, don't mark as loaded
    if (*targetWordCount == 0) {
        printf("[!] Warning: File loaded but no valid words found\n");
        *targetFileLoaded = false;
    } else {
        *targetFileLoaded = true;
        printf("Loaded %d tokens from File %d.\n", *targetWordCount, fileNumber);
    }
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
    for (int i = 0; text[i] && i < MAX_TEXT_LENGTH_ADV; i++) {
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
    if (!file1Loaded && !file2Loaded) {
        printf("No files loaded. Please use option 1 first to load files.\n");
        return;
    }

    printf("\n=== Advanced Text Analysis ===\n");
    
    int fileChoice;
    printf("Choose file to analyze:\n");
    if (file1Loaded) printf("1 - File 1 (%s)\n", inputFilePath1);
    if (file2Loaded) printf("2 - File 2 (%s)\n", inputFilePath2);
    printf("Enter choice: ");

    if (scanf("%d", &fileChoice) != 1) {
        int c; while ((c = getchar()) != '\n' && c != EOF);
        printf("Invalid number.\n");
        return;
    }
    int c; while ((c = getchar()) != '\n' && c != EOF);

    char (*selectedWords)[50] = NULL;
    int selectedWordCount = 0;
    char* selectedFilePath = NULL;

    if (fileChoice == 1 && file1Loaded) {
        selectedWords = words1;
        selectedWordCount = wordCount1;
        selectedFilePath = inputFilePath1;
    }
    else if (fileChoice == 2 && file2Loaded) {
        selectedWords = words2;
        selectedWordCount = wordCount2;
        selectedFilePath = inputFilePath2;
    }
    else {
        printf("Invalid file choice or file not loaded.\n");
        return;
    }

    printf("Analyzing: %s\n", selectedFilePath);
    printf("Total words: %d\n", selectedWordCount);

    // Calculate basic statistics
    int uniqueWords = 0;
    int totalChars = 0;
    char uniqueWordList[30000][50];
    int wordFreq[30000] = {0};

    for (int i = 0; i < selectedWordCount; i++) {
        totalChars += strlen(selectedWords[i]);
        
        // Count unique words and frequencies
        int found = 0;
        for (int j = 0; j < uniqueWords; j++) {
            if (strcmp(selectedWords[i], uniqueWordList[j]) == 0) {
                wordFreq[j]++;
                found = 1;
                break;
            }
        }
        if (!found && uniqueWords < 30000) {
            strcpy(uniqueWordList[uniqueWords], selectedWords[i]);
            wordFreq[uniqueWords] = 1;
            uniqueWords++;
        }
    }

    printf("\n=== ADVANCED ANALYSIS REPORT ===\n");
    printf("File: %s\n", selectedFilePath);
    printf("Total tokens: %d\n", selectedWordCount);
    printf("Unique words: %d\n", uniqueWords);
    printf("Total characters: %d\n", totalChars);

    if (selectedWordCount > 0) {
        printf("Average word length: %.1f characters\n", (float)totalChars / selectedWordCount);
        printf("Lexical diversity: %.3f\n", (float)uniqueWords / selectedWordCount);
    }

    // Show top 15 frequent words
    printf("\n--- TOP 15 FREQUENT WORDS ---\n");
    
    // Simple bubble sort for frequencies
    for (int i = 0; i < uniqueWords - 1; i++) {
        for (int j = 0; j < uniqueWords - i - 1; j++) {
            if (wordFreq[j] < wordFreq[j + 1]) {
                // Swap frequencies
                int tempFreq = wordFreq[j];
                wordFreq[j] = wordFreq[j + 1];
                wordFreq[j + 1] = tempFreq;
                
                // Swap words
                char tempWord[50];
                strcpy(tempWord, uniqueWordList[j]);
                strcpy(uniqueWordList[j], uniqueWordList[j + 1]);
                strcpy(uniqueWordList[j + 1], tempWord);
            }
        }
    }

    int topN = (uniqueWords < 15) ? uniqueWords : 15;
    for (int i = 0; i < topN; i++) {
        printf("%2d. %-20s %d occurrences\n", i + 1, uniqueWordList[i], wordFreq[i]);
    }

    // Word length analysis
    printf("\n--- WORD LENGTH DISTRIBUTION ---\n");
    int lengthCount[20] = {0}; // Count words of length 1-19+
    
    for (int i = 0; i < selectedWordCount; i++) {
        int len = strlen(selectedWords[i]);
        if (len >= 19) {
            lengthCount[19]++;
        } else if (len > 0) {
            lengthCount[len]++;
        }
    }

    for (int i = 1; i < 20; i++) {
        if (lengthCount[i] > 0) {
            if (i == 19) {
                printf("19+ chars: %d words (%.1f%%)\n", lengthCount[i], 
                       (float)lengthCount[i] / selectedWordCount * 100);
            } else {
                printf("%2d chars: %d words (%.1f%%)\n", i, lengthCount[i],
                       (float)lengthCount[i] / selectedWordCount * 100);
            }
        }
    }

    printf("\nAdvanced analysis completed successfully!\n");
}

// ====== 修改：排序和显示Top N单词 - 支持双文件 ======
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

    //simple sort: frequency descending, same frequency alphabetical
    for (int i = 0; i < ucnt - 1; i++) {
        for (int j = i + 1; j < ucnt; j++) {
            if (freq[j] > freq[i] || (freq[j] == freq[i] && strcmp(uniq[j], uniq[i]) < 0)) {
                int tf = freq[i]; freq[i] = freq[j]; freq[j] = tf;
                char tmp[50]; strcpy(tmp, uniq[i]); strcpy(uniq[i], uniq[j]); strcpy(uniq[j], tmp);
            }
        }
    }

    if (n > ucnt) n = ucnt;
    printf("Top %d words from selected file (%s):\n", n, isCSVFile(filePath) ? "CSV" : "Text");
    for (int i = 0; i < n; i++) {
        printf("%2d. %-20s %d\n", i + 1, uniq[i], freq[i]);
    }
}

// ====== 修改：保存结果到文件 - 支持双文件 ======
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

    // Ensure CSV extension
    char csv_path[512];
    strncpy(csv_path, path, sizeof(csv_path) - 1);
    csv_path[sizeof(csv_path) - 1] = '\0';
    
    // Replace or add .csv extension
    char* dot = strrchr(csv_path, '.');
    if (dot != NULL && (strcmp(dot, ".txt") == 0 || strcmp(dot, ".TXT") == 0)) {
        strcpy(dot, ".csv");
    } else if (dot == NULL) {
        strcat(csv_path, ".csv");
    }

#ifdef _WIN32
    FILE* f = fopen_u8(csv_path, "w");
#else
    FILE* f = fopen(csv_path, "w");
#endif
    if (!f) {
        printf("[!] Error: Cannot create output file '%s'\n", csv_path);
        printf("Recovery Guide:\n");
        printf("1. Make sure the file name is valid\n");
        printf("2. Check if you have write permissions in the target directory\n");
        printf("3. Ensure the directory exists\n");
        printf("4. Try a simpler file name in the current directory\n");
        printf("5. Examples: \"results.csv\", \"analysis_report.csv\"\n");
        printf("6. Close the file if it's already open in another program\n");
        perror("Detailed error");
        handleError("Cannot open output file");
        return;
    }

    // Load toxic words for analysis
    load_toxicwords();
    
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
    
    // Sort by frequency descending, then alphabetically
    for (int i = 0; i < ucnt - 1; i++) {
        for (int j = 0; j < ucnt - i - 1; j++) {
            if (freq[j] < freq[j + 1] || (freq[j] == freq[j + 1] && strcmp(uniq[j], uniq[j + 1]) > 0)) {
                int tf = freq[j]; freq[j] = freq[j + 1]; freq[j + 1] = tf;
                char tmp[50]; strcpy(tmp, uniq[j]); strcpy(uniq[j], uniq[j + 1]); strcpy(uniq[j + 1], tmp);
            }
        }
    }

    // ===== TOXIC WORDS ANALYSIS =====
    int toxic_words_count = 0;
    int total_toxic_occurrences = 0;
    char toxic_words_list[500][50];
    int toxic_freq[500] = {0};
    int toxic_severity[500] = {0};
    
    // Analyze toxic words
    for (int i = 0; i < ucnt; i++) {
        if (is_toxic_word(uniq[i])) {
            strcpy(toxic_words_list[toxic_words_count], uniq[i]);
            toxic_freq[toxic_words_count] = freq[i];
            toxic_severity[toxic_words_count] = get_toxic_severity(uniq[i]);
            total_toxic_occurrences += freq[i];
            toxic_words_count++;
            if (toxic_words_count >= 500) break;
        }
    }
    
    // Sort toxic words by frequency descending
    for (int i = 0; i < toxic_words_count - 1; i++) {
        for (int j = 0; j < toxic_words_count - i - 1; j++) {
            if (toxic_freq[j] < toxic_freq[j + 1]) {
                // Swap frequency
                int temp_freq = toxic_freq[j];
                toxic_freq[j] = toxic_freq[j + 1];
                toxic_freq[j + 1] = temp_freq;
                
                // Swap word
                char temp_word[50];
                strcpy(temp_word, toxic_words_list[j]);
                strcpy(toxic_words_list[j], toxic_words_list[j + 1]);
                strcpy(toxic_words_list[j + 1], temp_word);
                
                // Swap severity
                int temp_sev = toxic_severity[j];
                toxic_severity[j] = toxic_severity[j + 1];
                toxic_severity[j + 1] = temp_sev;
            }
        }
    }

    // Write CSV header
    fprintf(f, "Text Analysis Report\n");
    fprintf(f, "====================\n");
    fprintf(f, "Metric,Value\n");
    fprintf(f, "Total Tokens,%d\n", wordCount);
    fprintf(f, "Unique Words,%d\n", ucnt);
    fprintf(f, "Toxic Words Detected,%d\n", toxic_words_count);
    fprintf(f, "Total Toxic Occurrences,%d\n", total_toxic_occurrences);
    
    if (wordCount > 0) {
        float toxicity_percentage = (float)total_toxic_occurrences / wordCount * 100.0;
        fprintf(f, "Toxicity Percentage,%.2f%%\n", toxicity_percentage);
    } else {
        fprintf(f, "Toxicity Percentage,0.00%%\n");
    }
    fprintf(f, "\n");  // Empty line for separation
    
    // ===== TOXIC WORDS SECTION =====
    if (toxic_words_count > 0) {
        fprintf(f, "Toxic Words Analysis\n");
        fprintf(f, "Rank,Toxic Word,Frequency,Severity Level,Percentage of Total Words\n");
        
        for (int i = 0; i < toxic_words_count; i++) {
            float word_percentage = (float)toxic_freq[i] / wordCount * 100.0;
            fprintf(f, "%d,%s,%d,%d,%.2f%%\n", 
                   i + 1, 
                   toxic_words_list[i], 
                   toxic_freq[i], 
                   toxic_severity[i],
                   word_percentage);
        }
        fprintf(f, "\n");  // Empty line
    } else {
        fprintf(f, "Toxic Words Analysis\n");
        fprintf(f, "No toxic words detected in the text\n");
        fprintf(f, "\n");  // Empty line
    }
    
    // ===== WORD FREQUENCY DISTRIBUTION =====
    fprintf(f, "Word Frequency Distribution (Top 50)\n");
    fprintf(f, "Rank,Word,Frequency,Percentage,Is Toxic\n");
    
    // Calculate total for percentages
    int totalWords = wordCount;
    
    // Write word frequency data
    int topn = (ucnt < 50) ? ucnt : 50;
    for (int i = 0; i < topn; i++) {
        float percentage = (float)freq[i] / totalWords * 100.0;
        char is_toxic_flag[10] = "No";
        if (is_toxic_word(uniq[i])) {
            strcpy(is_toxic_flag, "Yes");
        }
        fprintf(f, "%d,%s,%d,%.2f%%,%s\n", i + 1, uniq[i], freq[i], percentage, is_toxic_flag);
    }
    
    // ===== SUMMARY STATISTICS =====
    fprintf(f, "\n");  // Empty line
    fprintf(f, "Summary Statistics\n");
    
    // Calculate additional statistics
    int singleOccurrence = 0;
    int highFrequency = 0; // words appearing 10+ times
    float avgFrequency = (float)totalWords / ucnt;
    
    for (int i = 0; i < ucnt; i++) {
        if (freq[i] == 1) singleOccurrence++;
        if (freq[i] >= 10) highFrequency++;
    }
    
    fprintf(f, "Metric,Value\n");
    fprintf(f, "Words with single occurrence,%d\n", singleOccurrence);
    fprintf(f, "Words with 10+ occurrences,%d\n", highFrequency);
    fprintf(f, "Average frequency per word,%.2f\n", avgFrequency);
    fprintf(f, "Lexical diversity ratio,%.3f\n", (float)ucnt / totalWords);
    
    // Toxic-specific statistics
    if (toxic_words_count > 0) {
        fprintf(f, "Most frequent toxic word,%s (%d occurrences)\n", 
               toxic_words_list[0], toxic_freq[0]);
        fprintf(f, "Highest severity toxic word,");
        
        // Find highest severity word
        int max_severity = 0;
        char highest_sev_word[50] = "";
        int highest_sev_freq = 0;
        for (int i = 0; i < toxic_words_count; i++) {
            if (toxic_severity[i] > max_severity) {
                max_severity = toxic_severity[i];
                strcpy(highest_sev_word, toxic_words_list[i]);
                highest_sev_freq = toxic_freq[i];
            }
        }
        fprintf(f, "%s (Level %d, %d occurrences)\n", highest_sev_word, max_severity, highest_sev_freq);
    }
    
    // ===== FILE INFORMATION =====
    fprintf(f, "\n");  // Empty line
    fprintf(f, "File Information\n");
    fprintf(f, "Metric,Value\n");
    fprintf(f, "Source File,%s\n", inputFilePath1);
    fprintf(f, "File Type,%s\n", isCSVFile(inputFilePath1) ? "CSV" : "Text");
    fprintf(f, "Analysis Date,%s\n", __DATE__);
    fprintf(f, "Toxic Dictionary Version,Loaded %d terms\n", g_toxic_count);
    
    fclose(f);
    
    // Console output summary
    printf("\nSaved results to %s\n", csv_path);
    printf("Your Report includes:\n");
    printf(" - Toxic words analysis (%d toxic words found)\n", toxic_words_count);
    printf(" - Word frequency distribution (top 50 words)\n");
    printf(" - Summary statistics\n");
    printf(" - File information\n");
}

bool isCSVFile(const char* filename) {
    const char* dot = strrchr(filename, '.');
    if (dot != NULL) {
        return (strcmp(dot, ".csv") == 0 || strcmp(dot, ".CSV") == 0);
    }
    return false;
}

// 处理CSV文件-将列转换为文本
void processCSVFile(FILE* f, char (*targetWords)[50], int* targetWordCount) {
    char line[4096];
    int columnCount = 0;
    char* columns[100]; // 假设最多100列

    printf("Processing your CSV file...\n");

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
    printf("Processed %d columns from CSV file\n", columnCount);
}

void handleFileMenu() {
    for (;;) {
        printf("\n=== File Management Menu ===\n");
        printf("1.1 - Load File 1 (Current: %s)\n", file1Loaded ? inputFilePath1 : "No file loaded");
        printf("1.2 - Load File 2 (Current: %s)\n", file2Loaded ? inputFilePath2 : "No file loaded");
        printf("1.3 - View file history of File 1\n");
        printf("1.4 - View file history of File 2\n");
        printf("1.5 - Exit to main menu\n");
        printf("Enter sub-choice (1.1 to 1.5): ");

        char subChoice[10];
        if (!read_line(subChoice, sizeof(subChoice))) {
            printf("Failed to read sub-choice.\n");
            continue;
        }

        if (strlen(subChoice)==0){
            printf("[X] Error. Please load a file.\n");
            continue;
        }
        
        if (strcmp(subChoice, "1.1") == 0) {
            printf("Enter file path for File 1 (supports .txt and .csv):\n> ");
            if (!read_line(inputFilePath1, sizeof(inputFilePath1))) {
                handleError("Failed to read file path");
                continue;
            }
            if (strlen(inputFilePath1) == 0) {
                printf("[X] Error. Please load a file!\n");
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
            if (strlen(inputFilePath2) == 0) {
                printf("No filename provided.\n");
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
            printf("Returning to main menu...\n");
            break;
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
    const char* originalType = isCSV ? "CSV" : "txt";
    const char* processedType = "txt"; // CSV files are always converted to text format

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

    // ⬇️ 每种算法单独统计
    SortStats sB, sQ, sM;

    stats_reset(); sort_pairs(a, n, key, ALG_BUBBLE); sB = g_stats;
    stats_reset(); sort_pairs(b, n, key, ALG_QUICK);  sQ = g_stats;
    stats_reset(); sort_pairs(c, n, key, ALG_MERGE);  sM = g_stats;

    if (topN > n) topN = n;

    printf("\n== Algorithm Comparison (key=%s, tiebreak=%s) ==\n",
        key == KEY_FREQ_DESC ? "freq desc" : "A→Z",
        g_use_secondary_tiebreak ? "ON" : "OFF");

    // ------- 三列排版：Bubble / Quick / Merge 并排显示 -------
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

    // 稳定性一致性检查（看前 cap 个是否完全一样）
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

    //  性能对比表（保持你原来的格式）
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

    // 也可计算"类型"层面的比例（unique）
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
    sort_pairs(arr, n, KEY_ALPHA, g_alg); // 按字母序排序

    const int perPage = 50;  // 每页显示多少个词，可以自己改
    int page = 0;            // 当前页，从 0 开始
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

        // 提示操作
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
        // 空输入：什么都不做，重新问
        if (cmd[0] == '\0') {
            continue;
        }

        // 严格匹配：只接受单个字符 n / p / q（大小写都可以）
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
            // 像 "npq"、"nn"、"x" 都会走到这里
            printf("[i] Unknown command. Please use n / p / q.\n");
        }
    }
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

void menu_sort_and_report(void) {
    int sub = -1;
    do {
        printf("\n=== Sorting & Reporting ===\n");
        printf("1) Set source file (current: %s)\n",
            g_use_file == 1 ? "File 1" : (g_use_file == 2 ? "File 2" : "Auto(File1->File2)"));
        printf("2) Set sort KEY (current: %s)\n",
            g_key == KEY_FREQ_DESC ? "Frequency in descending" : "Alphabetical (A->Z)");
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

// 只比较"主键"
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

void loadDictionaries() {
    printf("Dictionaries loaded (placeholder).\n");
}

void handleError(const char* message) {
    printf("Error: %s\n", message);
}

void displayNegativityScale() {
    printf("\nNegativity scale analysis (placeholder)\n");
}

void displayWordOccurrence() {
    printf("Word occurrence analysis (placeholder)\n");
}

void displayFileRecoveryTips(const char* problemType) {
    printf("\nFile Recovery Guide\n");

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
}
