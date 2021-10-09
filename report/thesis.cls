% Authors: Thomas Thüm (2018-02-13), modified by Jacob Krüger (2018-09-14), Elias Kuiter (2021-03-01)
% Tested with TeX Live 2020 on Windows, please report errors to the latest author.
\RequirePackage[l2tabu, orthodox]{nag} % shows warnings for common LaTeX code smells
\ProvidesClass{thesis}
\LoadClass[12pt,titlepage]{book}

% These packages are loaded by default. Some settings below depend on these packages (e.g. etoolbox for options), some are generally useful (e.g. microtype).
% The first few packages are must-haves:
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{geometry}
\usepackage{graphicx}
\usepackage{subfig}
\usepackage{makeidx}
\usepackage{microtype} % typographic adjustments
\usepackage[sort]{natbib} % bibliography
\usepackage{etoolbox} % toggle options
\usepackage{fancyhdr} % fancy page headers
\usepackage[table]{xcolor} % colors (also colored tables)
\usepackage{booktabs} % better tables
\usepackage{tabularx} % adds space-filling X column to tables
\usepackage{multirow} % span multiple rows with a single cell
\usepackage{hhline} % better horizontal lines in tables
\usepackage{mathtools} % \abs, \norm delimiter
\usepackage{amsthm} % theorem/proof environments
\usepackage{amssymb} % useful math symbols
\usepackage{wasysym} % more math symbols!
\usepackage{tcolorbox} % nice definitions, theorems etc.
\usepackage{subcaption} % captions for subfigures
\usepackage{silence} % filters unwanted LaTeX warnings
\usepackage{enumitem} % more options for \itemize
\usepackage{biolinum} % fancy sans-serif font for headings
\usepackage{marginnote} % put text in the margins
\usepackage{tikz} % THE package for drawing all sorts of things
\usepackage{listings} % used for code excerpts
\usepackage{float} % better figure positioning
\usepackage{makecell} % insert new line where usually impossible
\usepackage{adjustbox} % resize boxes easily
\usepackage{datatool} % read CSV automatically
\usepackage{pgf}
\usepackage{pgfplots}
\pgfplotsset{compat=1.15}
\newtoggle{final}
\newtoggle{print}
\newtoggle{german}
\DeclareOption{final}{\toggletrue{final}}
\DeclareOption{print}{\toggletrue{print}}
\DeclareOption{german}{\toggletrue{german}}
\ProcessOptions\relax
\newif\if@usepdf \@usepdffalse
\ifx\pdftexversion\@undefined
\else
\@usepdftrue
\fi

\iftoggle{final}{
	\newcommand{\todo}[1]{}
}{
	\newcommand{\todo}[1]{{\color{red} \emph{#1}}}
}

\iftoggle{german}{
	\usepackage[ngerman]{babel}
}{
	\usepackage[english]{babel}
}

\geometry{a4paper}
\geometry{left=30mm,right=30mm,top=30mm,bottom=25mm,head=14.5pt,marginparwidth=20mm,marginparsep=5mm}

\parindent 0cm
\parskip1.5ex plus0.5ex minus0.5ex

\setcounter{secnumdepth}{3}
\setcounter{tocdepth}{3}

\clubpenalty = 10000
\widowpenalty = 10000
\displaywidowpenalty = 10000

%\iftoggle{german}{
%	\labelformat{lstlisting}{Quelltext~#1}
%}{
%	\labelformat{lstlisting}{Listing~#1}
%}

\usepackage[notindex,nottoc]{tocbibind}
%\iftoggle{german}{
%	\renewcommand{\lstlistingname}{Quelltextverzeichnis}
%}{
%	\renewcommand{\lstlistingname}{List of Code Listings}
%}
%\renewcommand{\lstlistoflistings}{\begingroup
%	\tocfile{\lstlistingname}{lol}
%\endgroup}

\usepackage{url}
\expandafter\def\expandafter\UrlBreaks\expandafter{\UrlBreaks
  \do\a\do\b\do\c\do\d\do\e\do\f\do\g\do\h\do\i\do\j\do\k\do\l\do\m\do\n\do\o\do\p\do\q\do\r\do\s\do\t\do\u\do\v\do\w\do\x\do\y\do\z\do\A\do\B\do\C\do\D\do\E\do\F\do\G\do\H\do\I\do\J\do\K\do\L\do\M\do\N\do\O\do\P\do\Q\do\R\do\S\do\T\do\U\do\V\do\W\do\X\do\Y\do\Z\do\0\do\1\do\2\do\3\do\4\do\5\do\6\do\7\do\8\do\9}

\if@usepdf
 \usepackage{ae}
 \usepackage[raiselinks=true,
             bookmarks=true,
             bookmarksopenlevel=2,
             bookmarksopen=true,
             bookmarksnumbered=true,
						 pagebackref=true,
             hyperindex=true,
             \iftoggle{print}{hidelinks,}{colorlinks=true,}
						 linkcolor=pdflinkcolor,
						 anchorcolor=pdflinkcolor,
						 citecolor=pdfcitecolor,
						 filecolor=pdflinkcolor,
						 menucolor=pdflinkcolor,
						 urlcolor=pdflinkcolor,
             plainpages=false,
             pdfpagelabels=true,
						]{hyperref}
\else
  \usepackage{nohyperref}
\fi

\newif\if@usehyperref
\ifx\href\@undefined
\@usehyperreffalse
\else
\@usehyperreftrue
\fi

\iftoggle{german}{
	\renewcommand{\chapterautorefname}{Kapitel}
	\renewcommand{\sectionautorefname}{Abschnitt}
	\renewcommand{\subsectionautorefname}{Abschnitt}
}{
	\addto\extrasenglish{
    \renewcommand{\chapterautorefname}{Chapter}
    \renewcommand{\sectionautorefname}{Section}
    \renewcommand{\subsectionautorefname}{Section}
  }
}

\WarningFilter{glossaries}{Deprecated command}
\usepackage[nonumberlist,acronym,toc,nopostdot,nogroupskip]{glossaries}
\usepackage{morewrites}
\renewcommand{\glsnamefont}[1]{\textbf{#1}}
\setlength{\glsdescwidth}{0.8\textwidth}
\newglossarystyle{modsuper}{
  \glossarystyle{long}
  \renewcommand{\glsgroupskip}{}
  \renewcommand*{\glossaryentryfield}[5]{
    \glsentryitem{##1}\glstarget{##1}{\biolinum ##2} & ##3\glspostdescription\space ##5\\[3pt]}
}
\setglossarystyle{modsuper}

\newglossarystyle{supergroup}{
  \setglossarystyle{long}
  \renewcommand*{\glsgroupskip}{}
  \renewcommand{\glossentry}[2]{
    \tabularnewline
    \multicolumn{2}{l}{
     \bfseries\biolinum\glsentryitem{##1}\glstarget{##1}{\glossentryname{##1}}
    }
    \\[3pt]
  }
  \renewcommand{\subglossentry}[3]{
     \glssubentryitem{##2}
     \glstarget{##2}{\glossentryname{##2}}
     &
     \glossentrydesc{##2}\glspostdescription\space
     ##3\\[3pt]
  }
}

\newglossary[slg]{symbol}{sot}{stn}{\iftoggle{german}{Symbolverzeichnis}{List of Symbols}}

\newcommand{\headfont}{\biolinum}
\newcommand{\chapterheadfont}{\biolinum}

\pagestyle{fancy}
\renewcommand{\chaptermark}[1]{\markboth{\thechapter.\ #1}{}}
\fancyhf{}
\fancyhead[LE,RO]{{\headfont\thepage}}
\fancyhead[LO]{\headfont\nouppercase{\rightmark}}
\fancyhead[RE]{\headfont\nouppercase{\leftmark}}
\renewcommand{\headrulewidth}{0.5pt}
\renewcommand{\footrulewidth}{0pt}
\fancypagestyle{plain}{
	\fancyhf{}
	\renewcommand{\headrulewidth}{0pt}
	\renewcommand{\footrulewidth}{0pt}
}

\renewcommand{\chaptername}{}

\renewcommand{\section}{
\@startsection{section}
{1}
{0mm}
{1.5ex plus 1ex minus 1.2ex}
{0.5ex plus 0.5ex minus 0.5ex}
{\chapterheadfont\Large\bfseries}
}
\renewcommand{\subsection}{
\@startsection{subsection}
{2}
{0mm}
{1ex plus 1ex minus 1ex}
{0.3ex plus 0.3ex minus 0.3ex}
{\chapterheadfont\large\bfseries}
}
\renewcommand{\subsubsection}{
\@startsection{subsubsection}
{3}
{0mm}
{1ex plus 1ex minus 1ex}
{0.2ex plus 0.2ex minus 0.2ex}
{\chapterheadfont\normalsize\bfseries}
}
\renewcommand{\paragraph}{
\@startsection{paragraph}
{4}
{0mm}
{2ex plus 1ex minus 2ex}
{0.2ex plus 0.2ex minus 0.2ex}
{\chapterheadfont\normalsize\bfseries}
}

\newlength{\chapnolen}
\newlength{\chapparlen}
\newsavebox{\chapno}
\renewcommand{\@makechapterhead}[1]{
  \vspace*{0.2\textheight}
  \vskip 15\p@
  {\parindent \z@ \raggedright \normalfont
    \ifnum \c@secnumdepth >\m@ne
      \if@mainmatter
        \savebox{\chapno}{\chapterheadfont\huge\bfseries \thechapter.}
        \settowidth{\chapnolen}{\usebox{\chapno}}
        \parbox[t]{\chapnolen}{\usebox{\chapno}}\nobreak\leavevmode
      \fi
    \fi
    \interlinepenalty\@MM
    \setlength{\chapparlen}{\textwidth}
    \addtolength{\chapparlen}{-1.0\chapnolen}
    \addtolength{\chapparlen}{-2ex}
    \leavevmode\nobreak
    \parbox[t]{\chapparlen}{\raggedright\chapterheadfont\huge \bfseries #1\par\nobreak}
    \vskip 40\p@
  }}

\renewcommand{\@makeschapterhead}[1]{
  \vspace*{50\p@}
  {\parindent \z@ \raggedright
    \normalfont
    \interlinepenalty\@M
    \chapterheadfont \huge \bfseries  #1\par\nobreak
    \vskip 40\p@
  }
}


\let\@olddottedtocline\@dottedtocline
\renewcommand{\@dottedtocline}[5]{\@olddottedtocline{#1}{#2}{#3}{#4}{#5}}

\def\titlehead#1{\def\@titlehead{#1}}
\def\titlefoot#1{\def\@titlefoot{#1}}
\def\@titlehead{}
\def\@titlefoot{}

\def\ps@title{
\headheight 15mm
\footskip   0cm
\addtolength{\textheight}{-15mm}
\let\@mkboth\@gobbletwo
  \def\@oddhead{\vbox{\hbox to\textwidth{\@titlehead}
                \vskip 1.5mm
                \hbox to\textwidth{\hrulefill}}}
  \def\@oddfoot{\vbox{\vskip -1mm\hbox to\textwidth{\hrulefill}\vskip 1mm
                \hbox to\textwidth{\@titlefoot}}}
  \let\@evenhead\@oddhead
  \let\@evenfoot\@oddfoot
}

\renewenvironment{titlepage}
{
  \let\oldheadheight\headheight
  \let\oldfootskip\footskip
  \let\oldtextheight\textheight
  
   \cleardoublepage
   \if@twocolumn
      \@restonecoltrue\onecolumn
   \else
      \@restonecolfalse\newpage
    \fi
    \thispagestyle{title}
    \setcounter{page}\@ne
}
{\if@restonecol\twocolumn \else \newpage \fi
 \if@twoside\else
    \setcounter{page}\@ne
 \fi
 \let\headheight\oldheadheight
 \let\textheight\oldtextheight
 \let\footskip\oldfootskip
}

\renewcommand{\bibname}{Bibliography}

\renewcommand*{\backref}[1]{
}
\iftoggle{german}{
	\renewcommand*{\backrefalt}[4]{\footnotesize\color{gray}(zitiert auf Seite~#2)}
}{
	\renewcommand*{\backrefalt}[4]{\footnotesize\color{gray}(cited on Page~#2)}
}

%\iftoggle{german}{
%	\renewcommand{\bibname}{Literaturverzeichnis}
%    \bibliography{literature/thesis}
%
%}{
%	\renewcommand{\bibname}{Bibliography}
%    \bibliography{literature/thesis}
%
%}
\bibpunct{[}{]}{;}{a}{,}{,}

\renewenvironment{theindex}
{\if@twocolumn
   \@restonecolfalse
 \else
   \@restonecoltrue
 \fi
 \columnseprule \z@
 \columnsep 35\p@
 \twocolumn[\@makeschapterhead{\indexname}]
 \@mkboth{\indexname}
         {\indexname}
 \addcontentsline{toc}{chapter}{\indexname}
 \thispagestyle{fancy}
 \flushbottom
 \parindent\z@
 \parskip\z@ \@plus .3\p@\relax
 \let\item\@idxitem
 \def\,{\relax\ifmmode\mskip\thinmuskip
              \else\hskip0.2em\ignorespaces\fi}
 \raggedright}

\definecolor{red}{RGB}{220,0,0}
\definecolor{fin}{RGB}{0,105,180}
\definecolor{blue}{named}{fin}
\definecolor{green}{RGB}{73,125,0}
\definecolor{yellow}{RGB}{243,155,12}
\definecolor{pdflinkcolor}{RGB}{0,57,97}
\definecolor{pdfcitecolor}{RGB}{6,116,0}
\definecolor{light-gray}{gray}{0.9}
\definecolor{lightest-gray}{gray}{0.99}

\hfuzz=3.5pt
\vbadness=10000
\hbadness=10000

\DeclarePairedDelimiter\abs{\lvert}{\rvert}
\DeclarePairedDelimiter\norm{\lVert}{\rVert}

\let\oldabs\abs
\def\abs{\@ifstar{\oldabs}{\oldabs*}}
\let\oldnorm\norm
\def\norm{\@ifstar{\oldnorm}{\oldnorm*}}

\setitemize{itemsep=0pt,leftmargin=0.3in}
\setenumerate{itemsep=0pt,leftmargin=0.3in}
\renewcommand\labelitemi{\raisebox{0.08ex}{\small$\bullet$}}

\renewcommand*{\cleardoublepage}{\clearpage\if@twoside \ifodd\c@page\else
\hbox{}
\thispagestyle{empty}
\newpage
\if@twocolumn\hbox{}\newpage\fi\fi\fi}

\DeclareCaptionFont{bio}{\bfseries\biolinum}
\captionsetup{labelfont=bio}

\renewcommand\textbullet{\ensuremath{\bullet}}

\def\UrlFont{\rmfamily}
\newcommand{\blankpage}{\clearpage{\pagestyle{empty}\cleardoublepage}}

\tcbuselibrary{theorems,breakable,skins}
\newtcbtheorem[number within=chapter]{definition}{Definition}{enhanced,breakable,boxrule=0.3mm,arc=1mm,left=2mm,right=2mm,toptitle=1mm,bottomtitle=1mm,colback=lightest-gray,colframe=light-gray,coltitle=black,fonttitle=\bfseries\biolinum}{def}
\newtcbtheorem[number within=chapter]{theorem}{Theorem}{enhanced,breakable,boxrule=0.3mm,arc=1mm,left=2mm,right=2mm,toptitle=1mm,bottomtitle=1mm,colback=lightest-gray,colframe=light-gray,coltitle=black,fonttitle=\bfseries\biolinum}{th}
\newcommand\tcb@cnt@definitionautorefname{Definition}
\newcommand\tcb@cnt@theoremautorefname{Theorem}
\tcolorboxenvironment{proof}{breakable,blanker,left=4mm,before skip=10pt,after skip=20pt,borderline west={1.3mm}{0pt}{light-gray}}

\g@addto@macro\nag@captions{,phantomcaption}

\lstdefinelanguage{Java}{
  tabsize = 4,
  showstringspaces = false,
  commentstyle=\color[HTML]{839496},
  keywordstyle=\color[HTML]{71295b},
  stringstyle=\color[HTML]{2429c9},
  basicstyle = \footnotesize\ttfamily,
  breaklines = true,
  columns=fullflexible,
  keepspaces=true,
  sensitive=true,
  mathescape,
  morekeywords={class, int, boolean, void, normal_behavior, model_behavior, model, def, requires, requires_abs, ensures, ensures_abs, assignable, forall, exists, old, nothing, return}
}

\def\mydotfill{\leavevmode\xleaders\hb@xt@ .44em{\hss.\hss}\hfill\kern\z@}

\counterwithout{footnote}{chapter}