let SessionLoad = 1
let s:cpo_save=&cpo
set cpo&vim
inoremap <silent> <Plug>(ScrollViewToggle) <Cmd>ScrollViewToggle
inoremap <silent> <Plug>(ScrollViewRefresh) <Cmd>ScrollViewRefresh
inoremap <silent> <Plug>(ScrollViewPrev) <Cmd>ScrollViewPrev
inoremap <silent> <Plug>(ScrollViewNext) <Cmd>ScrollViewNext
inoremap <silent> <Plug>(ScrollViewLegend!) <Cmd>ScrollViewLegend!
inoremap <silent> <Plug>(ScrollViewLegend) <Cmd>ScrollViewLegend
inoremap <silent> <Plug>(ScrollViewLast) <Cmd>ScrollViewLast
inoremap <silent> <Plug>(ScrollViewFirst) <Cmd>ScrollViewFirst
inoremap <silent> <Plug>(ScrollViewEnable) <Cmd>ScrollViewEnable
inoremap <silent> <Plug>(ScrollViewDisable) <Cmd>ScrollViewDisable
cnoremap <silent> <Plug>(TelescopeFuzzyCommandSearch) e "lua require('telescope.builtin').command_history { default_text = [=[" . escape(getcmdline(), '"') . "]=] }"
inoremap <silent> <C-T> <Cmd>ToggleTerm
inoremap <silent> <expr> <BS> v:lua.MiniPairs.bs()
inoremap <Right> U<Right>
inoremap <Left> U<Left>
inoremap <nowait> <silent> <expr> <C-B> coc#float#has_scroll() ? "=coc#float#scroll(0)" : "<Left>"
inoremap <nowait> <silent> <expr> <C-F> coc#float#has_scroll() ? "=coc#float#scroll(1)" : "<Right>"
inoremap <silent> <expr> <C-Space> coc#refresh()
inoremap <C-J> <Plug>(coc-snippets-expand-jump)
inoremap <silent> <expr> <F13> coc#pum#visible() ? coc#pum#confirm() : "\<F13>"
inoremap <silent> <expr> <PageUp> coc#pum#visible() ? coc#pum#scroll(0) : "\<PageUp>"
inoremap <silent> <expr> <PageDown> coc#pum#visible() ? coc#pum#scroll(1) : "\<PageDown>"
inoremap <silent> <expr> <C-Y> coc#pum#visible() ? coc#pum#confirm() : coc#inline#visible() ? coc#inline#accept() :"\"
inoremap <silent> <expr> <C-E> coc#pum#visible() ? coc#pum#cancel() : coc#inline#visible() ? coc#inline#cancel() : "\"
inoremap <silent> <expr> <Up> coc#pum#visible() ? coc#pum#prev(0) : coc#inline#visible() ? coc#inline#prev() :"\<Up>"
inoremap <silent> <expr> <Down> coc#pum#visible() ? coc#pum#next(0) : coc#inline#visible() ? coc#inline#next() :"\<Down>"
inoremap <silent> <expr> <C-P> coc#pum#visible() ? coc#pum#prev(1) : coc#inline#visible() ? coc#inline#prev() : "\"
inoremap <silent> <expr> <C-N> coc#pum#visible() ? coc#pum#next(1) : coc#inline#visible() ? coc#inline#next() : "\"
inoremap <silent> <expr> <S-Tab> coc#pum#visible() ? coc#pum#prev(1) : "\"
inoremap <C-W> u
inoremap <C-U> u
vnoremap <nowait> <silent> <expr>  coc#float#has_scroll() ? coc#float#scroll(0) : ""
nnoremap <nowait> <silent> <expr>  coc#float#has_scroll() ? coc#float#scroll(0) : ""
vnoremap <nowait> <silent> <expr>  coc#float#has_scroll() ? coc#float#scroll(1) : ""
nnoremap <nowait> <silent> <expr>  coc#float#has_scroll() ? coc#float#scroll(1) : ""
snoremap <silent>  c
nnoremap  h
vnoremap <silent> 	 >gv
nnoremap <NL> j
nnoremap  k
nnoremap  l
snoremap  "_c
xnoremap <silent>  <Plug>(coc-range-select)
nnoremap <silent>  <Plug>(coc-range-select)
nmap  d
nnoremap <silent>  th <Cmd>ToggleTerm direction=horizontal
nnoremap <silent>  rt <Cmd>lua require("rust-tools").runnable.test()
nnoremap <silent>  rd <Cmd>lua require("rust-tools").debuggables.run()
nnoremap <silent>  rc <Cmd>lua require("rust-tools").crates.show_popup()
nnoremap <silent>  ls <Cmd>Telescope lsp_document_symbols
nnoremap <silent>  lr <Cmd>lua vim.lsp.buf.rename()
nnoremap <silent>  li <Cmd>Telescope lsp_implementations
nnoremap <silent>  ld <Cmd>Telescope lsp_definitions
nnoremap <silent>  la <Cmd>lua vim.lsp.buf.code_action()
nnoremap <silent>  lS <Cmd>Telescope lsp_dynamic_workspace_symbols
nnoremap <silent>  lR <Cmd>Telescope lsp_references
nnoremap <silent>  lD <Cmd>lua vim.lsp.buf.type_definition()
nnoremap <silent>  gu <Cmd>lua require("gitsigns").undo_stage_hunk()
nnoremap <silent>  gs <Cmd>Gitsigns stage_hunk
nnoremap <silent>  gr <Cmd>Gitsigns reset_hunk
nnoremap <silent>  gp <Cmd>lua require("gitsigns").preview_hunk()
nnoremap <silent>  gd <Cmd>lua require("gitsigns").diffthis()
nnoremap <silent>  gb <Cmd>lua require("gitsigns").blame_line({full=true})
nnoremap <silent>  gS <Cmd>lua require("gitsigns").stage_buffer()
nnoremap <silent>  gR <Cmd>lua require("gitsigns").reset_buffer()
nnoremap <silent>  fw <Cmd>Telescope grep_string
nnoremap <silent>  fs <Cmd>Telescope colorscheme
nnoremap <silent>  fk <Cmd>Telescope keymaps
nnoremap <silent>  fg <Cmd>Telescope live_grep
nnoremap <silent>  ff <Cmd>Telescope find_files
nnoremap <silent>  fd <Cmd>Telescope diagnostics
nnoremap <silent>  fc <Cmd>Telescope commands
nnoremap <silent>  fb <Cmd>Telescope buffers
nnoremap <silent>  fC <Cmd>Telescope command_history
nnoremap <silent>  cc <Cmd>lua require("Comment.api").toggle.linewise.current()
nnoremap <silent>  fr <Cmd>Telescope oldfiles
nnoremap <silent>  fh <Cmd>Telescope help_tags
nnoremap <silent>  tv <Cmd>ToggleTerm direction=vertical
vnoremap  s gs
nnoremap <silent>  sr cs
nnoremap <silent>  sd ds
nnoremap <silent>  sa ysa
vnoremap  / <Cmd>lua require("Comment.api").toggle.linewise(vim.fn.visualmode())
nnoremap <silent>  ih :CocCommand document.toggleInlayHint
nnoremap <nowait> <silent>  cl <Plug>(coc-codelens-action)
xnoremap <silent>  r <Plug>(coc-codeaction-refactor-selected)
nnoremap <silent>  re <Cmd>lua require("rust-tools").expand_macro.expand_macro()
nnoremap <nowait> <silent>  qf <Plug>(coc-fix-current)
nnoremap <nowait> <silent>  as <Plug>(coc-codeaction-source)
nnoremap <nowait> <silent>  ac <Plug>(coc-codeaction-cursor)
nnoremap <nowait> <silent>  a <Plug>(coc-codeaction-selected)
xnoremap <nowait> <silent>  a <Plug>(coc-codeaction-selected)
nnoremap <silent>  f <Plug>(coc-format-selected)
xnoremap <silent>  f <Plug>(coc-format-selected)
nnoremap <silent>  rn <Plug>(coc-rename)
nnoremap  oa <Cmd>OverseerTaskAction
nnoremap  or <Cmd>OverseerRun
nnoremap  ot <Cmd>OverseerToggle
nnoremap  rp :RunProject
nnoremap  rf :RunFile
nnoremap <silent>  rr <Cmd>lua require("rust-tools").runnable.run()
nnoremap <silent>  dB <Cmd>lua require("dap").set_breakpoint(vim.fn.input("Breakpoint condition: "))
nnoremap <silent>  db <Cmd>lua require("dap").toggle_breakpoint()
nnoremap  e :NvimTreeToggle
nnoremap  ts <Cmd>ToggleTerm direction=horizontal
nnoremap <silent>  tf <Cmd>ToggleTerm direction=float
nnoremap <silent>  tt <Cmd>ToggleTerm
nnoremap  s <Cmd>Telescope git_status
nnoremap <silent>  r <Plug>(coc-codeaction-refactor-selected)
nnoremap  h :Alpha
xnoremap <silent> # <Nop>
snoremap <silent> # <Nop>
omap <silent> % <Plug>(MatchitOperationForward)
xmap <silent> % <Plug>(MatchitVisualForward)
nmap <silent> % <Plug>(MatchitNormalForward)
nnoremap & :&&
xnoremap <silent> * <Nop>
snoremap <silent> * <Nop>
xnoremap <silent> <expr> @ mode() ==# 'V' ? ':normal! @'.getcharstr().'' : '@'
nnoremap <silent> K <Cmd>lua _G.show_docs()
xnoremap <silent> <expr> Q mode() ==# 'V' ? ':normal! @=reg_recorded()' : 'Q'
nnoremap Y y$
omap <silent> [% <Plug>(MatchitOperationMultiBackward)
xmap <silent> [% <Plug>(MatchitVisualMultiBackward)
nmap <silent> [% <Plug>(MatchitNormalMultiBackward)
onoremap <silent> [i <Cmd>lua MiniIndentscope.operator('top')
xnoremap <silent> [i <Cmd>lua MiniIndentscope.operator('top')
nnoremap <silent> [i <Cmd>lua MiniIndentscope.operator('top', true)
nnoremap <silent> [g <Plug>(coc-diagnostic-prev)
omap <silent> ]% <Plug>(MatchitOperationMultiForward)
xmap <silent> ]% <Plug>(MatchitVisualMultiForward)
nmap <silent> ]% <Plug>(MatchitNormalMultiForward)
onoremap <silent> ]i <Cmd>lua MiniIndentscope.operator('bottom')
xnoremap <silent> ]i <Cmd>lua MiniIndentscope.operator('bottom')
nnoremap <silent> ]i <Cmd>lua MiniIndentscope.operator('bottom', true)
nnoremap <silent> ]g <Plug>(coc-diagnostic-next)
xmap a% <Plug>(MatchitVisualTextObject)
onoremap <silent> ai <Cmd>lua MiniIndentscope.textobject(true)
xnoremap <silent> ai <Cmd>lua MiniIndentscope.textobject(true)
onoremap <nowait> <silent> ac <Plug>(coc-classobj-a)
xnoremap <nowait> <silent> ac <Plug>(coc-classobj-a)
onoremap <nowait> <silent> af <Plug>(coc-funcobj-a)
xnoremap <nowait> <silent> af <Plug>(coc-funcobj-a)
nnoremap cS <Plug>(nvim-surround-change-line)
omap <silent> g% <Plug>(MatchitOperationBackward)
xmap <silent> g% <Plug>(MatchitVisualBackward)
nmap <silent> g% <Plug>(MatchitNormalBackward)
nnoremap <silent> gr <Plug>(coc-references)
nnoremap <silent> gi <Plug>(coc-implementation)
nnoremap <silent> gy <Plug>(coc-type-definition)
nnoremap <silent> gd <Plug>(coc-definition)
onoremap <silent> gc <Cmd>lua MiniComment.textobject()
onoremap <silent> ii <Cmd>lua MiniIndentscope.textobject(false)
xnoremap <silent> ii <Cmd>lua MiniIndentscope.textobject(false)
onoremap <nowait> <silent> ic <Plug>(coc-classobj-i)
xnoremap <nowait> <silent> ic <Plug>(coc-classobj-i)
onoremap <nowait> <silent> if <Plug>(coc-funcobj-i)
xnoremap <nowait> <silent> if <Plug>(coc-funcobj-i)
nnoremap <silent> <expr> j v:count == 0 ? 'gj' : 'j'
nnoremap <silent> <expr> k v:count == 0 ? 'gk' : 'k'
xnoremap <silent> p "_dP
snoremap <silent> p "_dP
xnoremap s <Nop>
xnoremap <silent> sa :lua MiniSurround.add('visual')
nnoremap s <Nop>
noremap <silent> <Plug>(ScrollViewToggle) <Cmd>ScrollViewToggle
noremap <silent> <Plug>(ScrollViewRefresh) <Cmd>ScrollViewRefresh
noremap <silent> <Plug>(ScrollViewPrev) <Cmd>ScrollViewPrev
noremap <silent> <Plug>(ScrollViewNext) <Cmd>ScrollViewNext
noremap <silent> <Plug>(ScrollViewLegend!) <Cmd>ScrollViewLegend!
noremap <silent> <Plug>(ScrollViewLegend) <Cmd>ScrollViewLegend
noremap <silent> <Plug>(ScrollViewLast) <Cmd>ScrollViewLast
noremap <silent> <Plug>(ScrollViewFirst) <Cmd>ScrollViewFirst
noremap <silent> <Plug>(ScrollViewEnable) <Cmd>ScrollViewEnable
noremap <silent> <Plug>(ScrollViewDisable) <Cmd>ScrollViewDisable
nnoremap <silent> <F13>wv v
nnoremap <silent> <F13>ws s
nnoremap <silent> <F13>wq <Cmd>close
nnoremap <silent> <F13>we =
nnoremap <silent> <F13>w> >
nnoremap <silent> <F13>w< <
nnoremap <silent> <F13>w- -
nnoremap <silent> <F13>w+ +
nnoremap <silent> <F13>ct <Cmd>ClangdTypeHierarchy
nnoremap <silent> <F13>cs <Cmd>ClangdSymbolInfo
nnoremap <silent> <F13>cm <Cmd>ClangdMemoryUsage
nnoremap <silent> <F13>ci <Cmd>TSCppDefineClassFunc
nnoremap <silent> <F13>ch <Cmd>ClangdSwitchSourceHeader
nnoremap <silent> <F13>cI <Cmd>TSCppImplWrite
nnoremap <silent> <F13>bl <Cmd>buffers
nnoremap <silent> <F13>bc <Cmd>bdelete
snoremap <silent> <BS> c
snoremap <silent> <Del> c
snoremap <silent> <C-H> c
snoremap <C-R> "_c
xmap <silent> <Plug>(MatchitVisualTextObject) <Plug>(MatchitVisualMultiBackward)o<Plug>(MatchitVisualMultiForward)
onoremap <silent> <Plug>(MatchitOperationMultiForward) :call matchit#MultiMatch("W",  "o")
onoremap <silent> <Plug>(MatchitOperationMultiBackward) :call matchit#MultiMatch("bW", "o")
xnoremap <silent> <Plug>(MatchitVisualMultiForward) :call matchit#MultiMatch("W",  "n")m'gv``
xnoremap <silent> <Plug>(MatchitVisualMultiBackward) :call matchit#MultiMatch("bW", "n")m'gv``
nnoremap <silent> <Plug>(MatchitNormalMultiForward) :call matchit#MultiMatch("W",  "n")
nnoremap <silent> <Plug>(MatchitNormalMultiBackward) :call matchit#MultiMatch("bW", "n")
onoremap <silent> <Plug>(MatchitOperationBackward) :call matchit#Match_wrapper('',0,'o')
onoremap <silent> <Plug>(MatchitOperationForward) :call matchit#Match_wrapper('',1,'o')
xnoremap <silent> <Plug>(MatchitVisualBackward) :call matchit#Match_wrapper('',0,'v')m'gv``
xnoremap <silent> <Plug>(MatchitVisualForward) :call matchit#Match_wrapper('',1,'v'):if col("''") != col("$") | exe ":normal! m'" | endifgv``
nnoremap <silent> <Plug>(MatchitNormalBackward) :call matchit#Match_wrapper('',0,'n')
nnoremap <silent> <Plug>(MatchitNormalForward) :call matchit#Match_wrapper('',1,'n')
vnoremap <M-/> <Cmd>lua require("Comment.api").toggle.linewise(vim.fn.visualmode())
xnoremap <Plug>(comment_toggle_blockwise_visual) <Cmd>lua require("Comment.api").locked("toggle.blockwise")(vim.fn.visualmode())
xnoremap <Plug>(comment_toggle_linewise_visual) <Cmd>lua require("Comment.api").locked("toggle.linewise")(vim.fn.visualmode())
xnoremap <silent> <C-S> <Plug>(coc-range-select)
nnoremap <silent> <C-S> <Plug>(coc-range-select)
onoremap <silent> <Plug>(coc-classobj-a) :call CocAction('selectSymbolRange', v:false, '', ['Interface', 'Struct', 'Class'])
onoremap <silent> <Plug>(coc-classobj-i) :call CocAction('selectSymbolRange', v:true, '', ['Interface', 'Struct', 'Class'])
vnoremap <silent> <Plug>(coc-classobj-a) :call CocAction('selectSymbolRange', v:false, visualmode(), ['Interface', 'Struct', 'Class'])
vnoremap <silent> <Plug>(coc-classobj-i) :call CocAction('selectSymbolRange', v:true, visualmode(), ['Interface', 'Struct', 'Class'])
onoremap <silent> <Plug>(coc-funcobj-a) :call CocAction('selectSymbolRange', v:false, '', ['Method', 'Function'])
onoremap <silent> <Plug>(coc-funcobj-i) :call CocAction('selectSymbolRange', v:true, '', ['Method', 'Function'])
vnoremap <silent> <Plug>(coc-funcobj-a) :call CocAction('selectSymbolRange', v:false, visualmode(), ['Method', 'Function'])
vnoremap <silent> <Plug>(coc-funcobj-i) :call CocAction('selectSymbolRange', v:true, visualmode(), ['Method', 'Function'])
nnoremap <silent> <Plug>(coc-cursors-position) :call CocAction('cursorsSelect', bufnr('%'), 'position', 'n')
nnoremap <silent> <Plug>(coc-cursors-word) :call CocAction('cursorsSelect', bufnr('%'), 'word', 'n')
vnoremap <silent> <Plug>(coc-cursors-range) :call CocAction('cursorsSelect', bufnr('%'), 'range', visualmode())
nnoremap <silent> <Plug>(coc-refactor) :call       CocActionAsync('refactor')
nnoremap <silent> <Plug>(coc-command-repeat) :call       CocAction('repeatCommand')
nnoremap <silent> <Plug>(coc-float-jump) :call       coc#float#jump()
nnoremap <silent> <Plug>(coc-float-hide) :call       coc#float#close_all()
nnoremap <silent> <Plug>(coc-fix-current) :call       CocActionAsync('doQuickfix')
nnoremap <silent> <Plug>(coc-openlink) :call       CocActionAsync('openLink')
nnoremap <silent> <Plug>(coc-references-used) :call       CocActionAsync('jumpUsed')
nnoremap <silent> <Plug>(coc-references) :call       CocActionAsync('jumpReferences')
nnoremap <silent> <Plug>(coc-type-definition) :call       CocActionAsync('jumpTypeDefinition')
nnoremap <silent> <Plug>(coc-implementation) :call       CocActionAsync('jumpImplementation')
nnoremap <silent> <Plug>(coc-declaration) :call       CocActionAsync('jumpDeclaration')
nnoremap <silent> <Plug>(coc-definition) :call       CocActionAsync('jumpDefinition')
nnoremap <silent> <Plug>(coc-diagnostic-prev-error) :call       CocActionAsync('diagnosticPrevious', 'error')
nnoremap <silent> <Plug>(coc-diagnostic-next-error) :call       CocActionAsync('diagnosticNext',     'error')
nnoremap <silent> <Plug>(coc-diagnostic-prev) :call       CocActionAsync('diagnosticPrevious')
nnoremap <silent> <Plug>(coc-diagnostic-next) :call       CocActionAsync('diagnosticNext')
nnoremap <silent> <Plug>(coc-diagnostic-info) :call       CocActionAsync('diagnosticInfo')
nnoremap <silent> <Plug>(coc-format) :call       CocActionAsync('format')
nnoremap <silent> <Plug>(coc-rename) :call       CocActionAsync('rename')
nnoremap <Plug>(coc-codeaction-source) :call       CocActionAsync('codeAction', '', ['source'], v:true)
nnoremap <Plug>(coc-codeaction-refactor) :call       CocActionAsync('codeAction', 'cursor', ['refactor'], v:true)
nnoremap <Plug>(coc-codeaction-cursor) :call       CocActionAsync('codeAction', 'cursor')
nnoremap <Plug>(coc-codeaction-line) :call       CocActionAsync('codeAction', 'currline')
nnoremap <Plug>(coc-codeaction) :call       CocActionAsync('codeAction', '')
vnoremap <Plug>(coc-codeaction-refactor-selected) :call       CocActionAsync('codeAction', visualmode(), ['refactor'], v:true)
vnoremap <silent> <Plug>(coc-codeaction-selected) :call       CocActionAsync('codeAction', visualmode())
vnoremap <silent> <Plug>(coc-format-selected) :call       CocActionAsync('formatSelected', visualmode())
nnoremap <Plug>(coc-codelens-action) :call       CocActionAsync('codeLensAction')
nnoremap <Plug>(coc-range-select) :call       CocActionAsync('rangeSelect',     '', v:true)
vnoremap <silent> <Plug>(coc-range-select-backward) :call       CocActionAsync('rangeSelect',     visualmode(), v:false)
vnoremap <silent> <Plug>(coc-range-select) :call       CocActionAsync('rangeSelect',     visualmode(), v:true)
nnoremap <Plug>PlenaryTestFile :lua require('plenary.test_harness').test_file(vim.fn.expand("%:p"))
tnoremap <F13>: :
nnoremap <C-K> k
nnoremap <C-J> j
nnoremap <C-H> h
nnoremap <C-F13> w
nnoremap <silent> <F13>g <Cmd>Telescope live_grep
nnoremap <silent> <F13>f <Cmd>Telescope find_files
nnoremap <silent> <F13>p <Cmd>bprevious
nnoremap <silent> <F13>n <Cmd>bnext
nnoremap <silent> <F13>l l
nnoremap <silent> <F13>k k
nnoremap <silent> <F13>j j
nnoremap <silent> <F13>h h
nmap <C-W><C-D> d
vnoremap <silent> <S-Tab> <gv
nnoremap <C-L> l
inoremap <nowait> <silent> <expr>  coc#float#has_scroll() ? "=coc#float#scroll(0)" : "<Left>"
inoremap <silent> <expr>  coc#pum#visible() ? coc#pum#cancel() : coc#inline#visible() ? coc#inline#cancel() : "\"
inoremap <nowait> <silent> <expr>  coc#float#has_scroll() ? "=coc#float#scroll(1)" : "<Right>"
inoremap <silent> <expr> 	 coc#pum#visible() ? coc#pum#next(1) : v:lua.check_back_space() ? "	" : coc#refresh()
inoremap <NL> <Plug>(coc-snippets-expand-jump)
inoremap <expr>  v:lua.require'nvim-autopairs'.completion_confirm()
inoremap <silent> <expr>  coc#pum#visible() ? coc#pum#next(1) : coc#inline#visible() ? coc#inline#next() : "\"
inoremap <silent> <expr>  coc#pum#visible() ? coc#pum#prev(1) : coc#inline#visible() ? coc#inline#prev() : "\"
inoremap <silent>  <Cmd>ToggleTerm
inoremap  u
inoremap  u
inoremap <silent> <expr>  coc#pum#visible() ? coc#pum#confirm() : coc#inline#visible() ? coc#inline#accept() :"\"
inoremap <expr> " v:lua.MiniPairs.closeopen('""', "[^\\].")
inoremap <expr> ' v:lua.MiniPairs.closeopen("''", "[^%a\\].")
inoremap <expr> ( v:lua.MiniPairs.open("()", "[^\\].")
inoremap <expr> ) v:lua.MiniPairs.close("()", "[^\\].")
inoremap <expr> [ v:lua.MiniPairs.open("[]", "[^\\].")
inoremap <expr> ] v:lua.MiniPairs.close("[]", "[^\\].")
inoremap <expr> ` v:lua.MiniPairs.closeopen("``", "[^\\].")
inoremap <expr> { v:lua.MiniPairs.open("{}", "[^\\].")
inoremap <expr> } v:lua.MiniPairs.close("{}", "[^\\].")
let &cpo=s:cpo_save
unlet s:cpo_save
set cindent
set clipboard=unnamedplus
set completeopt=menu,menuone,noselect
set noemoji
set eventignore=CursorHoldI,CursorHold
set fileformats=unix,dos,mac
set fillchars=horiz:���,horizdown:┬,horizup:┴,vert:\ ,verthoriz:┼,vertleft:┤,vertright:├
set grepformat=%f:%l:%c:%m
set grepprg=rg\ --vimgrep\ -uu\ 
set guifont=Fira\ Code,Symbols_Nerd_Font_Mono:h14
set helplang=en
set nohlsearch
set ignorecase
set laststatus=3
set noloadplugins
set mouse=a
set operatorfunc=<SNR>17_FormatFromSelected
set packpath=/opt/homebrew/Cellar/neovim/0.11.4/share/nvim/runtime
set runtimepath=~/.config/nvim,~/.local/share/nvim/site,~/.local/share/nvim/lazy/lazy.nvim,~/.local/share/nvim/lazy/nvim-autopairs,~/.local/share/nvim/lazy/neoscroll.nvim,~/.local/share/nvim/lazy/nvim-scrollview,~/.local/share/nvim/lazy/mini.animate,~/.local/share/nvim/lazy/cord.nvim,~/.local/share/nvim/lazy/crates.nvim,~/.local/share/nvim/lazy/which-key.nvim,~/.local/share/nvim/lazy/telescope-frecency.nvim,~/.local/share/nvim/lazy/sqlite.lua,~/.local/share/nvim/lazy/nvim-neoclip.lua,~/.local/share/nvim/lazy/telescope-ui-select.nvim,~/.local/share/nvim/lazy/telescope-fzf-native.nvim,~/.local/share/nvim/lazy/telescope.nvim,~/.local/share/nvim/lazy/vim-fugitive,~/.local/share/nvim/lazy/code_runner.nvim,~/.local/share/nvim/lazy/toggleterm.nvim,~/.local/share/nvim/lazy/mini.nvim,~/.local/share/nvim/lazy/neogen,~/.local/share/nvim/lazy/vim-sleuth,~/.local/share/nvim/lazy/multicursor.nvim,~/.local/share/nvim/lazy/nvim-notify,~/.local/share/nvim/lazy/neotest-rust,~/.local/share/nvim/lazy/FixCursorHold.nvim,~/.local/share/nvim/lazy/neotest,~/.local/share/nvim/lazy/conform.nvim,~/.local/share/nvim/lazy/plenary.nvim,~/.local/share/nvim/lazy/refactoring.nvim,~/.local/share/nvim/lazy/coc.nvim,~/.local/share/nvim/lazy/nvim-treesitter,~/.local/share/nvim/lazy/Comment.nvim,~/.local/share/nvim/lazy/nvim-lspconfig,~/.local/share/nvim/lazy/rust-tools.nvim,~/.local/share/nvim/lazy/telescope-dap.nvim,~/.local/share/nvim/lazy/nvim-dap-virtual-text,~/.local/share/nvim/lazy/nvim-nio,~/.local/share/nvim/lazy/nvim-dap-ui,~/.local/share/nvim/lazy/nvim-dap,~/.local/share/nvim/lazy/overseer.nvim,~/.local/share/nvim/lazy/alpha-nvim,~/.local/share/nvim/lazy/nvim-surround,~/.local/share/nvim/lazy/vim-rhubarb,~/.local/share/nvim/lazy/nvim-web-devicons,~/.local/share/nvim/lazy/nvim-tree.lua,~/.local/share/nvim/lazy/sonokai,/opt/homebrew/Cellar/neovim/0.11.4/share/nvim/runtime,/opt/homebrew/Cellar/neovim/0.11.4/share/nvim/runtime/pack/dist/opt/netrw,/opt/homebrew/Cellar/neovim/0.11.4/share/nvim/runtime/pack/dist/opt/matchit,/opt/homebrew/Cellar/neovim/0.11.4/lib/nvim,~/.local/state/nvim/lazy/readme,~/.local/share/nvim/lazy/sonokai/after
set scrolloff=8
set sessionoptions=blank,buffers,curdir,folds,help,options,tabpages,winsize,terminal
set shortmess=OoTtFClI
set showbreak=↪\ \ \ 
set noshowmode
set sidescrolloff=8
set smartcase
set smartindent
set softtabstop=8
set statusline=%{%(nvim_get_current_win()==#g:actual_curwin\ ||\ &laststatus==3)\ ?\ v:lua.MiniStatusline.active()\ :\ v:lua.MiniStatusline.inactive()%}
set noswapfile
set termguicolors
set timeoutlen=300
set undofile
set updatetime=300
set window=58
set nowritebackup
let s:so_save = &g:so | let s:siso_save = &g:siso | setg so=0 siso=0 | setl so=-1 siso=-1
let v:this_session=expand("<sfile>:p")
silent only
silent tabonly
cd /Volumes/Synthesis/workspace
if expand('%') == '' && !&modified && line('$') <= 1 && getline(1) == ''
  let s:wipebuf = bufnr('%')
endif
if &shortmess =~ 'A'
  set shortmess=aoOA
else
  set shortmess=aoO
endif
badd +32 ~/.zshrc
badd +43 engine/src/lib.rs
badd +28 ~/.config/nvim/coc-settings.json
badd +11 engine/src/ui3d.msl
badd +33 ~/.config/nvim/lua/plugins/animations.lua
badd +18 engine/Cargo.toml
badd +1 .cargo/config.toml
badd +1 engine/engine.xcodeproj/project.xcworkspace/xcuserdata/solmidnight.xcuserdatad/UserInterfaceState.xcuserstate
argglobal
%argdel
$argadd ~/.zshrc
edit engine/src/ui3d.msl
wincmd t
let s:save_winminheight = &winminheight
let s:save_winminwidth = &winminwidth
set winminheight=0
set winheight=1
set winminwidth=0
set winwidth=1
argglobal
balt engine/engine.xcodeproj/project.xcworkspace/xcuserdata/solmidnight.xcuserdatad/UserInterfaceState.xcuserstate
let s:cpo_save=&cpo
set cpo&vim
inoremap <buffer> <M-e> l<Cmd>lua require('nvim-autopairs.fastwrap').show()
let &cpo=s:cpo_save
unlet s:cpo_save
setlocal keymap=
setlocal noarabic
setlocal autoindent
setlocal nobinary
set breakindent
setlocal breakindent
setlocal breakindentopt=
setlocal bufhidden=
setlocal buflisted
setlocal buftype=
setlocal cindent
setlocal cinkeys=0{,0},0),0],:,0#,!^F,o,O,e
setlocal cinoptions=
setlocal cinscopedecls=public,protected,private
setlocal cinwords=if,else,while,do,for,switch
setlocal colorcolumn=
setlocal comments=:#
setlocal commentstring=#\ %s
setlocal complete=.,w,b,u,t
setlocal completefunc=
setlocal completeslash=
setlocal concealcursor=
setlocal conceallevel=0
setlocal nocopyindent
setlocal nocursorbind
setlocal nocursorcolumn
setlocal nocursorline
setlocal cursorlineopt=both
setlocal nodiff
setlocal eventignorewin=
setlocal expandtab
if &filetype != 'conf'
setlocal filetype=conf
endif
setlocal fixendofline
setlocal foldcolumn=0
setlocal foldenable
setlocal foldexpr=0
setlocal foldignore=#
setlocal foldlevel=0
setlocal foldmarker={{{,}}}
setlocal foldmethod=manual
setlocal foldminlines=1
setlocal foldnestmax=20
setlocal foldtext=foldtext()
setlocal formatexpr=
setlocal formatlistpat=^\\s*\\d\\+[\\]:.)}\\t\ ]\\s*
setlocal formatoptions=jcroql
setlocal iminsert=0
setlocal imsearch=-1
setlocal includeexpr=
setlocal indentexpr=
setlocal indentkeys=0{,0},0),0],:,0#,!^F,o,O,e
setlocal noinfercase
setlocal iskeyword=@,48-57,_,192-255
set linebreak
setlocal linebreak
setlocal nolisp
setlocal lispoptions=
setlocal nolist
setlocal matchpairs=(:),{:},[:]
setlocal modeline
setlocal modifiable
setlocal nrformats=bin,hex
set number
setlocal number
setlocal numberwidth=4
setlocal omnifunc=
setlocal nopreserveindent
setlocal nopreviewwindow
setlocal quoteescape=\\
setlocal noreadonly
set relativenumber
setlocal relativenumber
setlocal norightleft
setlocal rightleftcmd=search
setlocal scrollback=-1
setlocal noscrollbind
setlocal shiftwidth=4
set signcolumn=yes
setlocal signcolumn=yes
setlocal smartindent
setlocal nosmoothscroll
setlocal softtabstop=-1
setlocal nospell
setlocal spellcapcheck=[.?!]\\_[\\])'\"\\t\ ]\\+
setlocal spellfile=
setlocal spelllang=en
setlocal spelloptions=
setlocal statuscolumn=
setlocal suffixesadd=
setlocal noswapfile
setlocal synmaxcol=3000
if &syntax != 'conf'
setlocal syntax=conf
endif
setlocal tabstop=8
setlocal tagfunc=
setlocal textwidth=0
setlocal undofile
setlocal varsofttabstop=
setlocal vartabstop=
setlocal winblend=0
setlocal nowinfixbuf
setlocal nowinfixheight
setlocal nowinfixwidth
setlocal winhighlight=
setlocal wrap
setlocal wrapmargin=0
silent! normal! zE
let &fdl = &fdl
let s:l = 11 - ((10 * winheight(0) + 28) / 57)
if s:l < 1 | let s:l = 1 | endif
keepjumps exe s:l
normal! zt
keepjumps 11
normal! 0
tabnext 1
if exists('s:wipebuf') && len(win_findbuf(s:wipebuf)) == 0 && getbufvar(s:wipebuf, '&buftype') isnot# 'terminal'
  silent exe 'bwipe ' . s:wipebuf
endif
unlet! s:wipebuf
set winheight=1 winwidth=20
set shortmess=OoTtFClI
let &winminheight = s:save_winminheight
let &winminwidth = s:save_winminwidth
let s:sx = expand("<sfile>:p:r")."x.vim"
if filereadable(s:sx)
  exe "source " . fnameescape(s:sx)
endif
let &g:so = s:so_save | let &g:siso = s:siso_save
doautoall SessionLoadPost
unlet SessionLoad
" vim: set ft=vim :
