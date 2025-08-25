% ====================================================================
% PART 1: 文档类声明、选项与核心宏包
% ====================================================================

% --- 1. 文档类声明与标识 ---
\NeedsTeXFormat{LaTeX2e}[1995/12/01]
\ProvidesClass{cumcmthesis}
              [2024/05/21 v3.1 Custom Thesis Template] % <- 已更新描述

% --- 2. 选项定义与处理 ---
\newif\if@mcm@bwprint % 创建一个 if-else 开关，用于控制彩色/黑白打印

\DeclareOption{colorprint}{\@mcm@bwprintfalse} % 定义 colorprint 选项
\DeclareOption{bwprint}{\@mcm@bwprinttrue}     % 定义 bwprint 选项
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}} % 将未知选项传递给 article

\ExecuteOptions{colorprint} % 设置默认选项为彩色打印
\ProcessOptions\relax       % 正式处理所有选项

% --- 3. 基础类与核心宏包加载 ---
\LoadClass[a4paper,12pt]{article} % 加载 article 作为模板的基础

\RequirePackage{ifxetex} % 判断是否使用 XeTeX 编译
\RequireXeTeX           % 强制要求使用 XeTeX
\ifxetex\else           % 如果不是 XeTeX，则报错并停止
\ClassError{mcmthesis}{You must use the `xelatex' driver\MessageBreak Please choose `xelatex'}{%
Just choose `xelatex', no `pdflatex' or `latex' and so on.}
\fi

\RequirePackage{ctex}             % 中文支持核心宏包
\RequirePackage{geometry}         % 页面布局和边距设置
\RequirePackage{amsmath}
\RequirePackage[dvipsnames, svgnames, x11names]{xcolor} % 颜色支持
\RequirePackage{graphicx}         % 插入图片
\RequirePackage{float}            % 浮动体（图、表）控制
\RequirePackage{cprotect}         % 保护脆弱命令
\RequirePackage{url}              % URL链接处理
\RequirePackage{indentfirst}      % 首行缩进
\RequirePackage{caption}          % 图表标题格式定制
\RequirePackage{enumitem}         % 列表环境定制
\RequirePackage{ulem}             % 下划线等文本装饰
\RequirePackage{calc}             % 在LaTeX命令中进行计算
\RequirePackage[titletoc,title]{appendix} % 附录功能增强
\RequirePackage{etoolbox}         % 提供编程工具钩子
\RequirePackage{hyperref}         % 超链接
\RequirePackage{cleveref}         % 智能交叉引用


% --- 数学、表格、子图等专业宏包 ---

\RequirePackage{amsfonts}
\RequirePackage{amssymb}
\RequirePackage{bm}
\RequirePackage{array}
\RequirePackage{longtable}
\RequirePackage{booktabs,tabularx}
\RequirePackage{multirow}
\RequirePackage{bigstrut}
\RequirePackage{bigdelim}
\RequirePackage{subcaption}
\RequirePackage{listings}
\usepackage{algorithm}             % 算法环境
\usepackage{algorithmicx}
\usepackage{algpseudocode}


% ====================================================================
% PART 2: PAGE LAYOUT, FONTS, AND GENERAL FORMATTING
% ====================================================================

% --- 1. 页面布局与通用格式 ---
\geometry{top=25mm,bottom=25mm,left=25mm,right=25mm} % 页面边距
\renewcommand*{\baselinestretch}{1.38} % 全局行间距
\setlength\parindent{2em} % 全局段首缩进2个汉字宽度

\let\mcm@oldtabular\tabular % 重定义 tabular 环境，使其内部行距与正文一致
\let\mcm@endoldtabular\endtabular
\renewenvironment{tabular}%
	{\bgroup%
	\renewcommand{\arraystretch}{1.38}%
	\mcm@oldtabular}%
	{\mcm@endoldtabular\egroup}

\newcolumntype{Y}{>{\centering\arraybackslash}X} % 为 tabularx 定义新的居中列类型
\newcolumntype{C}{>{\centering\arraybackslash}X} % 定义 C 列类型
\newcolumntype{R}{>{\raggedleft\arraybackslash}X} % 定义 R 列类型
\newcolumntype{L}{>{\raggedright\arraybackslash}X} % 定义 L 列类型
\renewcommand\normalsize{% % 微调 CTeX 默认的 normalsize (12pt)
	\@setfontsize\normalsize{12.05}{14.45}%
	\abovedisplayskip 12\p@ \@plus3\p@ \@minus7\p@
	\abovedisplayshortskip \z@ \@plus3\p@
	\belowdisplayshortskip 6.5\p@ \@plus3.5\p@ \@minus3\p@}

% --- 2. 字体与代码环境设置 ---
\RequirePackage{fontspec}
\setmainfont{TeX Gyre Termes}  % 主字体 (西文)
\setsansfont{Arial}           % 无衬线字体
\setCJKfamilyfont{kai}[AutoFakeBold]{simkai.ttf} % 定义楷体字体族
\newcommand*{\kai}{\CJKfamily{kai}}              % 定义 \kai 命令
\setCJKfamilyfont{song}[AutoFakeBold]{SimSun}     % 定义宋体字体族
\newcommand*{\song}{\CJKfamily{song}}            % 定义 \song 命令

\newfontfamily\yaheiconsola{YaHei.Consolas.1.11b.ttf} % 定义代码专用字体
\setmonofont[
Contextuals={Alternate},
ItalicFont = Fira Code Retina Nerd Font Complete.otf % 避免字体警告
]{YaHei.Consolas.1.11b.ttf}


\definecolor{codegreen}{rgb}{0,0.6,0} % 定义代码高亮颜色
\lstset{ % 设置 listings 代码环境的全局样式
    tabsize=4, captionpos=b, numbers=left, numbersep=1em,
    sensitive=true, showtabs=false, frame=shadowbox, breaklines=true,
    keepspaces=true, showspaces=false, showstringspaces=false,
    breakatwhitespace=false, basicstyle=\yaheiconsola, keywordstyle=\color{NavyBlue},
    commentstyle=\color{codegreen}, numberstyle=\color{gray},
    stringstyle=\color{PineGreen!90!black}, rulesepcolor=\color{red!20!green!20!blue!20}
}

% ====================================================================
% PART 3:章节、目录与标题格式
% ====================================================================
\RequirePackage{titlesec}
\RequirePackage{titletoc}

% --- (A) 目录条目格式控制 (使用 titletoc) ---
\titlecontents{section}[0em]
  {\songti\zihao{-4}}
  {\thecontentslabel\ }
  {}
  {\hspace{.5em}\titlerule*[4pt]{$\cdot$}\contentspage}

\titlecontents{subsection}[2em]
  {\vspace{0.1\baselineskip}\songti\zihao{-4}}
  {\thecontentslabel\ }
  {}
  {\hspace{.5em}\titlerule*[4pt]{$\cdot$}\contentspage}

\titlecontents{subsubsection}[4em]
  {\vspace{0.1\baselineskip}\songti\zihao{-4}}
  {\thecontentslabel\ }
  {}
  {\hspace{.5em}\titlerule*[4pt]{$\cdot$}\contentspage}

% --- (B) 正文章节编号与标题格式控制 ---
\renewcommand\thesection{\chinese{section}、}
\renewcommand\thesubsection{\arabic{section}.\arabic{subsection}}
\renewcommand\thesubsubsection{\thesubsection.\arabic{subsubsection}}
\setcounter{secnumdepth}{3} % 目录和编号深度到 subsubsection

% --- Section (一级标题) ---
% 使用三号字，居中，粗体，黑体
\titleformat{\section}
  {\centering\bfseries\songti\zihao{3}}
  {\thesection}
  {1em}
  {}

% --- Subsection (二级标题) ---
% 使用四号字，左对齐，粗体，黑体
\titleformat{\subsection}
  {\bfseries\heiti\zihao{4}}
  {\thesubsection}
  {0.8em}
  {}

% --- Subsubsection (三级标题) ---
% 使用小四号字，左对齐，宋体 (不加粗，以形成对比)
\titleformat{\subsubsection}
  {\bfseries\songti\zihao{-4}}
  {\thesubsubsection}
  {0.8em}
  {}
% (可选) 如果需要，还可以用 \titlespacing 调整标题的垂直间距
% \titlespacing*{\section}{0pt}{3.5ex plus 1ex minus .2ex}{2.3ex plus .2ex}
% \titlespacing*{\subsection}{0pt}{3.25ex plus 1ex minus .2ex}{1.5ex plus .2ex}

% ====================================================================
% PART 4:自定义封面与摘要
% ====================================================================

% --- 1. 自定义封面 (maketitle) ---
\newcommand{\studentid}[1]{\def\@studentid{#1}}
\newcommand{\studentname}[1]{\def\@studentname{#1}}
\newcommand{\college}[1]{\def\@college{#1}}
\newcommand{\major}[1]{\def\@major{#1}}
\def\@studentid{}
\def\@studentname{}
\def\@college{}
\def\@major{}

\renewcommand{\maketitle}{
  \begin{titlepage}
    \begin{center}
      \vspace*{2cm}
      {\huge\bfseries \@title\par}
      \vspace{2cm}
      \begin{tabular}{rl}
        \Large\textbf{学 \quad 号：} & \Large\underline{\makebox[8cm]{\@studentid}} \\[12pt]
        \Large\textbf{姓 \quad 名：} & \Large\underline{\makebox[8cm]{\@studentname}} \\[12pt]
        \Large\textbf{所在学院：} & \Large\underline{\makebox[8cm]{\@college}} \\[12pt]
        \Large\textbf{专 \quad 业：} & \Large\underline{\makebox[8cm]{\@major}} \\[12pt]
      \end{tabular}
    \end{center}
    \thispagestyle{empty} % 封面页不显示页码
  \end{titlepage}
}

% --- 2. 自定义摘要 (abstract & keywords) ---
\newcommand{\mcm@keywords@content}{}
\newcommand\keywords[1]{%
  \renewcommand{\mcm@keywords@content}{#1}%
}
\renewenvironment{abstract}{%
  \begin{center}%
    {\zihao{4}\bfseries \abstractname\vspace{-.5em}\vspace{\z@}}%
  \end{center}%
  \quotation
}{%
  \endquotation
  \ifx\mcm@keywords@content\empty\else
    \vspace{\fill}
    \par\noindent
    {\zihao{-4}\heiti\bfseries\mcm@cap@keywordsname：}%
    \mcm@keywords@content
  \fi
  \clearpage
}
\renewenvironment{quotation}
	{\list{}{\listparindent 2em%
	 \itemindent \listparindent
	 \rightmargin\z@ \leftmargin\z@ \parsep \z@ \@plus\p@}%
	\item\relax}
	{\endlist}

% ====================================================================
% PART 5: ENVIRONMENTS AND LOCALIZATION (CHINESE TEXT)
% ====================================================================

% --- 1. 汉化文本常量定义 (供后续命令调用) ---
\newcommand*{\mcm@cap@definition}{定义}
\newcommand*{\mcm@cap@theorem}{定理}
\newcommand*{\mcm@cap@lemma}{引理}
\newcommand*{\mcm@cap@corollary}{推论}
\newcommand*{\mcm@cap@assumption}{假设}
\newcommand*{\mcm@cap@conjecture}{猜想}
\newcommand*{\mcm@cap@axiom}{公理}
\newcommand*{\mcm@cap@principle}{定律}
\newcommand*{\mcm@cap@problem}{问题}
\newcommand*{\mcm@cap@example}{例}
\newcommand*{\mcm@cap@proof}{证明}
\newcommand*{\mcm@cap@solution}{解}
\newcommand*{\mcm@cap@contentsname}{目录}
\newcommand*{\mcm@cap@listfigurename}{插图清单}
\newcommand*{\mcm@cap@listtablename}{附表清单}
\newcommand*{\mcm@cap@refname}{参考文献}
\newcommand*{\mcm@cap@indexname}{索引}
\newcommand*{\mcm@cap@figurename}{图}
\newcommand*{\mcm@cap@tablename}{表}
\newcommand*{\mcm@cap@appendixname}{附录}
\newcommand*{\mcm@cap@abstractname}{摘要}
\newcommand*{\mcm@cap@keywordsname}{关键字}

% --- 2. 使用上面的常量来汉化标准标题 ---
\renewcommand\contentsname{\mcm@cap@contentsname}
\renewcommand\listfigurename{\mcm@cap@listfigurename}
\renewcommand\listtablename{\mcm@cap@listtablename}
\renewcommand\refname{\mcm@cap@refname}
\renewcommand\indexname{\mcm@cap@indexname}
\renewcommand\figurename{\mcm@cap@figurename}
\renewcommand\tablename{\mcm@cap@tablename}
\renewcommand\appendixname{\mcm@cap@appendixname}
\renewcommand\abstractname{\mcm@cap@abstractname}

% --- 3. 定理类环境定义 ---
\newtheorem{definition}{\mcm@cap@definition}
\newtheorem{theorem}{\mcm@cap@theorem}
\newtheorem{lemma}{\mcm@cap@lemma}
\newtheorem{corollary}{\mcm@cap@corollary}
\newtheorem{assumption}{\mcm@cap@assumption}
\newtheorem{conjecture}{\mcm@cap@conjecture}
\newtheorem{axiom}{\mcm@cap@axiom}
\newtheorem{principle}{\mcm@cap@principle}
\newtheorem{problem}{\mcm@cap@problem}
\newtheorem{example}{\mcm@cap@example}
\newtheorem{proof}{\mcm@cap@proof}
\newtheorem{solution}{\mcm@cap@solution}

% ====================================================================
% PART 6: 引用、链接与最终工具
% ====================================================================

% --- 1. 图、表、列表等环境定制 ---
\renewcommand*{\textfraction}{0.05}
\renewcommand*{\topfraction}{0.9}
\renewcommand*{\bottomfraction}{0.8}
\renewcommand*{\floatpagefraction}{0.85}
\DeclareGraphicsExtensions{.pdf,.eps,.jpg,.png}
\graphicspath{{figures/}{figure/}{pictures/}{images/}}
\DeclareCaptionFont{song}{\songti}
\DeclareCaptionFont{minusfour}{\zihao{-4}}
\captionsetup[figure]{format=hang, labelsep=quad, font={song,minusfour,bf}, position=bottom}
\captionsetup[table]{format=hang, labelsep=quad, font={song,minusfour,bf}, position=top}
\setlist{topsep=0.3em, itemsep=0ex plus 0.1ex, leftmargin=1.5em, labelsep=0.5em, labelwidth=2em}

% --- 2. 超链接与交叉引用 (Hyperref & Cleveref) ---
\hypersetup{
    pdfstartview=FitH, CJKbookmarks=true, bookmarksnumbered=true,
    bookmarksopen=true, colorlinks, pdfborder=001,
    allcolors=black, breaklinks=true
}
\if@mcm@bwprint \AtBeginDocument{\hypersetup{hidelinks}} \else\relax\fi
\pdfstringdefDisableCommands{\def\cftdotfill{ }}

% --- 完整的 Cleveref 汉化设置 ---
\crefformat{figure}{#2图~#1#3}
\crefrangeformat{figure}{图~(#3#1#4)\;\~{}\;(#5#2#6)}
\crefmultiformat{figure}{图~(#2#1#3)}{和~(#2#1#3)}{，(#2#1#3)}{和~(#2#1#3)}
\crefformat{table}{#2表#1#3}
\crefrangeformat{table}{表(#3#1#4)\;\~{}\;(#5#2#6)}
\crefmultiformat{table}{表~(#2#1#3)}{和~(#2#1#3)}{，(#2#1#3)}{和~(#2#1#3)}
\crefformat{equation}{#2~(#1#3)}
\crefrangeformat{equation}{~(#3#1#4)\;\~{}\;(#5#2#6)}
\crefmultiformat{equation}{~(#2#1#3)}{ 和~(#2#1#3)}{，(#2#1#3)}{ 和~(#2#1#3)}
\crefformat{definition}{#2\mcm@cap@definition~#1#3}
\crefrangeformat{definition}{\mcm@cap@definition~(#3#1#4)\;\~{}\;(#5#2#6)}
\crefmultiformat{definition}{\mcm@cap@definition~(#2#1#3)}{ 和~(#2#1#3)}{，(#2#1#3)}{ 和~(#2#1#3)}
\crefformat{theorem}{#2\mcm@cap@theorem~#1#3}
\crefrangeformat{theorem}{\mcm@cap@theorem~(#3#1#4)\;\~{}\;(#5#2#6)}
\crefmultiformat{theorem}{\mcm@cap@theorem~(#2#1#3)}{ 和~(#2#1#3)}{，(#2#1#3)}{ 和~(#2#1#3)}
\crefformat{lemma}{#2\mcm@cap@lemma~#1#3}
\crefrangeformat{lemma}{\mcm@cap@lemma~(#3#1#4)\;\~{}\;(#5#2#6)}
\crefmultiformat{lemma}{\mcm@cap@lemma~(#2#1#3)}{ 和~(#2#1#3)}{，(#2#1#3)}{ 和~(#2#1#3)}
\crefformat{corollary}{#2\mcm@cap@corollary~#1#3}
\crefrangeformat{corollary}{\mcm@cap@corollary~(#3#1#4)\;\~{}\;(#5#2#6)}
\crefmultiformat{corollary}{\mcm@cap@corollary~(#2#1#3)}{ 和~(#2#1#3)}{，(#2#1#3)}{ 和~(#2#1#3)}
\crefformat{assumption}{#2\mcm@cap@assumption~#1#3}
\crefrangeformat{assumption}{\mcm@cap@assumption~(#3#1#4)\;\~{}\;(#5#2#6)}
\crefmultiformat{assumption}{\mcm@cap@assumption~(#2#1#3)}{ 和~(#2#1#3)}{，(#2#1#3)}{ 和~(#2#1#3)}

% --- 3. 参考文献环境与辅助命令 ---
\AtBeginEnvironment{thebibliography}{\phantomsection \addcontentsline{toc}{section}{\refname}}
\renewenvironment{thebibliography}[1]
     {\section*{\refname}%
      \@mkboth{\MakeUppercase\refname}{\MakeUppercase\refname}%
      \list{\@biblabel{\@arabic\c@enumiv}}%
           {\settowidth\labelwidth{\@biblabel{#1}}%
            \leftmargin\labelwidth \advance\leftmargin\labelsep \@openbib@code
            \usecounter{enumiv} \let\p@enumiv\@empty \renewcommand\theenumiv{\@arabic\c@enumiv}}%
      \sloppy \clubpenalty4000 \@clubpenalty \clubpenalty \widowpenalty4000%
      \sfcode`\.\@m}
     {\def\@noitemerr{\@latex@warning{Empty `thebibliography' environment}}\endlist}

\newcommand{\upcite}[1]{\textsuperscript{\textsuperscript{\cite{#1}}}}
\newcommand{\supercite}[1]{\textsuperscript{\textsuperscript{\cite{#1}}}}


% --- 文件结束 ---
\endinput