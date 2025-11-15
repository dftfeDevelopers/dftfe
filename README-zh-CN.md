# DFT-FE：基于有限元的密度泛函理论

关于
----
DFT-FE 是一个用于从头算（第一性原理）材料建模的 C++ 代码，基于 Kohn-Sham 密度泛函理论。DFT-FE 的起源可追溯到 [Computational Materials Physics Group](http://www-perso[...])。

DFT-FE 基于自适应有限元离散化，能够在相同框架下处理赝势（pseudopotential）和全电子（all-electron）计算，并且集成了可扩展且高效的求解器以满足规模化计算的需求。[...]


安装说明
---------
DFT-FE 代码在许多与有限元、几何、网格等相关的部分依赖于 deal.II 库，并且通过 deal.II 依赖于 p4est 用于并行自适应网格处理。
安装所需依赖项和 DFT-FE 本身的步骤已在 DFT-FE 手册的 “Installation”（安装）一节中描述（在此处下载开发版手册 [链接](https://github.com/df[...] )）。

我们为不同机器上的 DFT-FE 开发分支（`publicGithubDevelop`）创建了若干基于 shell 的安装脚本：
  - [OLCF Frontier](https://github.com/dftfeDevelopers/install_DFTFE/tree/frontierDevelop)
  - [NERSC Perlmutter](https://github.com/dftfeDevelopers/install_DFTFE/tree/perlmutterDevelop)
  - [ALCF Polaris](https://github.com/dftfeDevelopers/install_DFTFE/tree/polarisScript)
  - [UMICH Greatlakes](https://github.com/dftfeDevelopers/install_DFTFE/tree/greatlakesDevelop)
    


运行 DFT-FE
-----------
关于如何运行 DFT-FE（包括示例演示） 的说明也可以在手册的 “Running DFT-FE”（运行 DFT-FE）一节中找到（在此处下载开发版手册 [链接](https://github.com/dftfeDevelo[...])）。


为 DFT-FE 做贡献
----------------
想了解更多关于为 DFT-FE 开发做贡献的信息，请参见此处：[Contributing](https://github.com/dftfeDevelopers/dftfe/wiki/Contributing)。


更多信息
--------

 - 有关代码功能、引用方式、致谢以及 DFT-FE 相关的新闻，请参阅官方 [网站](https://sites.google.com/umich.edu/dftfe)。
  
 - 查阅由 Doxygen 生成的 [文档](https://dftfedevelopers.github.io/dftfe/)。

 - 有关 DFT-FE 的问题、安装、BUG 等，请使用 [DFT-FE 讨论论坛](https://groups.google.com/forum/#!forum/dftfe-user-group)。 

 - 有关最新新闻、更新和版本发布的消息，请发送邮件至 dft-fe.admin@umich.edu，我们会将您加入公告邮件列表。
 
 - DFT-FE 主要基于 [deal.II 库](http://www.dealii.org/)。如果您对 deal.II 有具体问题，请使用 [deal.II 讨论论坛](https://www.dealii.org/mail.html)。
 
 - 如果您有不适合公开或归档邮件列表讨论的 DFT-FE 相关问题，可以联系以下人员：
    - Phani Motamarri: phanim@iisc.ac.in
    - Sambit Das: dsambit@umich.edu
    - Vikram Gavini: vikramg@umich.edu 

 - 下列人员在过去或当前对推进 DFT-FE 的目标做出了重要贡献：（以下列表按姓氏字母顺序排列）
   - 导师 / 开发负责人
      - Dr. Sambit Das（美国密歇根大学安娜堡分校）
      - Prof. Vikram Gavini（美国密歇根大学安娜堡分校）
      - Prof. Phani Motamarri（印度科学研究所，Indian Institute of Science）
   - 主要开发者  
      - Dr. Sambit Das（美国密歇根大学安娜堡分校）
      - Prof. Phani Motamarri（印度科学研究所，Indian Institute of Science）
      - Nikhil Kodali（印度科学研究所，Indian Institute of Science）    
      - Kartick Ramakrishnan（印度科学研究所，Indian Institute of Science）

 - 完整的为 DFT-FE 做出贡献的作者名单可以在 [authors](authors) 中找到。    

许可证
-------
DFT-FE 在 [LGPL v2.1 或更高版本](https://github.com/dftfeDevelopers/dftfe/blob/publicGithubDevelop/LICENSE) 许可下发布。
