# Convert every .docx in submit/ to PDF with Microsoft Word (run after build_m2_docs.py).
$sub = Join-Path (Split-Path $PSScriptRoot -Parent) "submit"
$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
try {
    Get-ChildItem -Path $sub -Filter *.docx | Where-Object { $_.Name -notlike '~$*' } | ForEach-Object {
        $doc = $word.Documents.Open($_.FullName, $false, $true)
        $pdf = [IO.Path]::ChangeExtension($_.FullName, ".pdf")
        $doc.SaveAs2($pdf, 17)
        "{0}: {1} pages" -f (Split-Path $pdf -Leaf), $doc.ComputeStatistics(2)
        $doc.Close(0)
    }
} finally { $word.Quit() }
