#! /usr/local/bin/perl

use strict;
# use Encode;
# global variable
# my $paper_num = 0;

my $result_pmid_title = "pmid_title.csv";
open(OUTPUT1, ">$result_pmid_title") || die "Cannot open file $result_pmid_title for writing\n";

my $result_pmids = "pmids.csv";
open(OUTPUT2, ">$result_pmids") || die "Cannot open file $result_pmids for writing\n";

my $result_citations = "citations.csv";
open(OUTPUT3, ">$result_citations") || die "Cannot open file $result_citations for writing\n";
# my $num_paper = "num_paper.csv";
# open(OUTPUT3, ">$num_paper") || die "Cannot open file $num_paper for writing\n";
my $file_num = 0;
my $paper_considered = 0;

use Cwd; # module for finding the current working directory

sub ScanDirectory {

   # get the start directory from the passed argument
   my ($workdir) = shift;

   # keep track of where we began
   my ($startdir) = &cwd;

   # change directory to start directory
   chdir($workdir) or die "Unable to enter dir $workdir:$!\n";

   # read directory and store filenames in names arry
   opendir(DIR, ".") or die "Unable to open $workdir:$!\n";
   my @names = readdir(DIR) or die "Unable to read $workdir:$!\n";
   closedir(DIR);


   foreach my $name (@names){
      my $pmid;
      my $title = '';
      my @ref_pmids = ();
      my @ref_titles = ();
      my @refs = ();
      my @refs_with_pmid =();
      my $ref_pmid;
      my $ref_title;
      next if ($name eq ".");
      next if ($name eq "..");

      # if directory, recurse
      if (-d $name) {

         &ScanDirectory($name);
         next;
      }

      my $path = Cwd::cwd();
      my $filename = $path."/".$name;
      my $myfile;
      open ($myfile, $name);
      $file_num++;
      local $/=undef;
      my $scalar = <$myfile>;
      close ($name);

      # find lines in file matching pattern
      #while (<$myfile>) {

      $pmid = $1 if($scalar =~ m/<article-id pub-id-type="pmid">[^\d]*(\d+)[^\d]*?<\/article-id>/);
      $title = $1 if($scalar =~ m/<article-title>([\s\S]*?)<\/article-title>/);
      $title =~ s/[\s]/ /g;   #remove all \r and \n and \t, white spaces in-between the title string
      
      while ($scalar =~ m/(<ref id=[\s\S]*?<\/ref>)/g){
         push @refs, $1;
      }
      
      #}

      for (my $k = 0;$k <= $#refs;$k++)
      {
         if($refs[$k] =~ m/<pub-id pub-id-type="pmid">/){
            push @refs_with_pmid, $refs[$k];
         }

      }

         
      for (my $k = 0;$k <= $#refs_with_pmid;$k++)
      {
         while($refs_with_pmid[$k] =~ m/<article-title>([\s\S]*?)<\/article-title>[\s\S]*?<pub-id pub-id-type="pmid">[^\d]*(\d+)[^\d]*?<\/pub-id>/g){           
            $ref_title = $1;
            $ref_pmid = $2;
            $ref_title =~ s/[\s]/ /g;

            push @ref_pmids, $ref_pmid;
            push @ref_titles, $ref_title;
         }
      }


      if($pmid && $title && $ref_pmids[0] && ($#ref_pmids == $#ref_titles)){
         $paper_considered++;
         print OUTPUT1 "$pmid,$title\n";
         print OUTPUT2 "$pmid\n";
         for (my $k = 0;$k <= $#ref_pmids;$k++)
         {  
            if($ref_titles[$k]){
               print OUTPUT1 "$ref_pmids[$k],$ref_titles[$k]\n";
               print OUTPUT2 "$ref_pmids[$k]\n"; 
               print OUTPUT3 "$pmid,$ref_pmids[$k]\n"; 
            }
            else{
               my $path = Cwd::cwd();
               print "$path\n";
               print "$name\n"; 
            }
         }
      }
      elsif($pmid && !$title){
         print "$pmid\n";
         my $path = Cwd::cwd();
         print "$path\n";
         print "$name\n"; 
      }
     
   }

   chdir($startdir) or die "Unable to change to dir $startdir:$!\n";
}

# main program

# initialize start directory to current directory
my $mystartdir = ".";

&ScanDirectory($mystartdir);
print "Total number of papers: $file_num\n";
print "Number of papers have pimd and at least one citation with pmid: $paper_considered\n";
# print OUTPUT "Number of files: $file_num";
